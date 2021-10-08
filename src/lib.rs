use std::cell::Cell;
use std::convert::{Infallible, TryFrom};
use std::ffi::{CString, OsString};
use std::mem::MaybeUninit;
use std::ops::ControlFlow;
use std::os::raw::c_int;
use std::os::unix::ffi::OsStringExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};

use nix::errno::Errno;
use nix::fcntl::{open, OFlag};
use nix::libc;
use nix::pty::{grantpt, posix_openpt, ptsname, unlockpt, PtyMaster};
use nix::sys::select::{pselect, FdSet};
use nix::sys::signal::{kill, sigprocmask, SigSet, SigmaskHow, Signal};
use nix::sys::signal::{sigaction, SaFlags, SigAction, SigHandler};
use nix::sys::stat::Mode;
use nix::sys::termios::{tcgetattr, tcsetattr, SetArg, Termios};
use nix::sys::wait::{waitpid, WaitStatus};
use nix::unistd::{close, dup2, execvp, fork, isatty, read, setsid, write};
use nix::unistd::{ForkResult, Pid};
use nix::NixPath;

mod cfmakeraw;
use cfmakeraw::cfmakeraw;

fn tiocgwinsz(fd: RawFd) -> nix::Result<libc::winsize> {
    nix::ioctl_read_bad!(ioctl, libc::TIOCGWINSZ, libc::winsize);
    let mut size = MaybeUninit::uninit();
    unsafe { ioctl(fd, size.as_mut_ptr()) }?;
    Ok(unsafe { size.assume_init() })
}

fn tiocswinsz(fd: RawFd, size: &libc::winsize) -> nix::Result<()> {
    nix::ioctl_write_ptr_bad!(ioctl, libc::TIOCSWINSZ, libc::winsize);
    unsafe { ioctl(fd, size as *const _) }.map(|_| ())
}

thread_local! {
    static ORIG_TERM_ATTRS: Cell<Option<Termios>> = Cell::new(None);
    static CHILD_PID: Cell<Option<Pid>> = Cell::new(None);
}

fn install_exit_handler() -> Result<(), Error> {
    extern "C" fn handle_exit() {
        ORIG_TERM_ATTRS.with(|attrs| {
            if let Some(attrs) = attrs.take() {
                let _ = tcsetattr(0, SetArg::TCSANOW, &attrs);
            }
        });
    }

    if unsafe { libc::atexit(handle_exit) } != 0 {
        return Err(Error {
            kind: ExitHandlerSetupFailed,
            call: Some("atexit".into()),
            errno: None,
        });
    }
    Ok(())
}

fn install_terminate_handler() -> Result<(), Error> {
    const SIGNALS: [Signal; 3] =
        [Signal::SIGHUP, Signal::SIGINT, Signal::SIGTERM];

    extern "C" fn handle_terminate(_: c_int) {
        CHILD_PID.with(|pid| {
            if let Some(pid) = pid.take() {
                let _ = kill(pid, Signal::SIGHUP);
            }
        });
        exit(0);
    }

    let action = SigAction::new(
        SigHandler::Handler(handle_terminate),
        SaFlags::empty(),
        SigSet::empty(),
    );

    for signal in SIGNALS {
        unsafe { sigaction(signal, &action) }
            .map_err(SignalSetupFailed.with("sigaction"))?;
    }
    Ok(())
}

fn child_exec<Arg, Path>(
    args: impl IntoIterator<Item = Arg>,
    tty_name: &Path,
    attrs: &Termios,
    winsize: &libc::winsize,
) -> Result<Infallible, Error>
where
    Arg: Into<OsString>,
    Path: NixPath + ?Sized,
{
    setsid().map_err(ChildSetupFailed.with("setsid"))?;
    for fd in 0..=2 {
        let _ = close(fd);
    }

    let tty_fd = open(tty_name, OFlag::O_RDWR, Mode::empty())
        .map_err(ChildOpenTtyFailed.with("open"))?;

    for fd in 0..=2 {
        let _ = dup2(tty_fd, fd);
    }

    if tty_fd > 2 {
        let _ = close(tty_fd);
    }

    tcsetattr(0, SetArg::TCSANOW, attrs).map_err(
        SetAttrFailed {
            target: Child,
            caller: Child,
        }
        .with("tcsetattr"),
    )?;
    tiocswinsz(0, winsize).map_err(
        SetSizeFailed {
            target: Child,
            caller: Child,
        }
        .with(Ioctl("TIOCSWINSZ")),
    )?;

    let args: Vec<_> = args
        .into_iter()
        .map(|a| CString::new(a.into().into_vec()).unwrap())
        .collect();
    execvp(&args[0], &args).map_err(ChildExecFailed.with("execvp"))?;
    unreachable!();
}

struct Buffers<const N: usize> {
    pub input: [u8; N],
    pub output: Vec<u8>,
}

fn handle_stdin_ready<Fh: FilterHooks, const N: usize>(
    pty: &PtyMaster,
    filter: &mut Fh,
    bufs: &mut Buffers<N>,
) -> Result<ControlFlow<()>, Error> {
    let nread = match read(0, &mut bufs.input) {
        Ok(0) => {
            return Err(Error {
                kind: EmptyParentRead,
                call: Some("read".into()),
                errno: None,
            });
        }
        r => r.map_err(ParentReadFailed.with("read"))?,
    };

    let inbuf = &bufs.input;
    let mut write_err = false;
    chunked(
        &mut bufs.output,
        |c| filter.on_parent_data(&inbuf[..nread], |data| c.add(data)),
        |chunk| {
            if !write_err && write(pty.as_raw_fd(), chunk).is_err() {
                write_err = true;
            }
        },
    );

    Ok(if write_err {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    })
}

fn handle_pty_ready<Fh: FilterHooks, const N: usize>(
    pty: &PtyMaster,
    filter: &mut Fh,
    bufs: &mut Buffers<N>,
) -> Result<ControlFlow<()>, Error> {
    let nread = match read(pty.as_raw_fd(), &mut bufs.input) {
        Ok(0) | Err(_) => return Ok(ControlFlow::Break(())),
        Ok(n) => n,
    };

    let inbuf = &bufs.input;
    let mut write_err = None;
    chunked(
        &mut bufs.output,
        |c| filter.on_child_data(&inbuf[..nread], |data| c.add(data)),
        |chunk| {
            if write_err.is_none() {
                write_err = write(0, chunk).err();
            }
        },
    );
    write_err
        .map(ParentWriteFailed.with("write"))
        .map_or(Ok(ControlFlow::Continue(())), Err)
}

static SIGWINCH_RECEIVED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigwinch(_: c_int) {
    SIGWINCH_RECEIVED.store(true, Ordering::Relaxed);
}

fn update_child_winsize(pty: &PtyMaster) -> Result<(), Error> {
    if !SIGWINCH_RECEIVED.swap(false, Ordering::Relaxed) {
        return Ok(());
    }

    let size = tiocgwinsz(0).map_err(GetSizeFailed.with("tiocgwinsz"))?;
    tiocswinsz(pty.as_raw_fd(), &size).map_err(
        SetSizeFailed {
            target: Child,
            caller: Parent,
        }
        .with(Ioctl("TIOCSWINSZ")),
    )?;
    Ok(())
}

pub struct Error {
    pub kind: ErrorKind,
    pub call: Option<CallName>,
    pub errno: Option<Errno>,
}

#[derive(Clone, Copy)]
pub enum Client {
    Parent,
    Child,
}

use Client::*;

#[non_exhaustive]
pub enum ErrorKind {
    #[non_exhaustive]
    NotATty,
    #[non_exhaustive]
    GetAttrFailed,
    #[non_exhaustive]
    SetAttrFailed {
        target: Client,
        caller: Client,
    },
    #[non_exhaustive]
    GetSizeFailed,
    #[non_exhaustive]
    SetSizeFailed {
        target: Client,
        caller: Client,
    },
    #[non_exhaustive]
    CreatePtyFailed,
    #[non_exhaustive]
    CreateChildFailed,
    #[non_exhaustive]
    SignalSetupFailed,
    #[non_exhaustive]
    ExitHandlerSetupFailed,
    #[non_exhaustive]
    ChildCommFailed,
    #[non_exhaustive]
    ChildSetupFailed,
    #[non_exhaustive]
    ChildOpenTtyFailed,
    #[non_exhaustive]
    ChildExecFailed,
    #[non_exhaustive]
    EmptyParentRead,
    #[non_exhaustive]
    ParentReadFailed,
    #[non_exhaustive]
    ParentWriteFailed,
    #[non_exhaustive]
    GetChildStatusFailed,
    UnexpectedChildStatus(UnexpectedChildStatus),
}

use ErrorKind::*;
pub struct UnexpectedChildStatus(WaitStatus);

impl ErrorKind {
    fn with(self, name: impl Into<CallName>) -> impl FnOnce(Errno) -> Error {
        let name = name.into();
        |errno| Error {
            kind: self,
            call: Some(name),
            errno: Some(errno),
        }
    }
}

impl Error {
    fn with_kind(kind: impl Into<ErrorKind>) -> Self {
        Self {
            kind: kind.into(),
            call: None,
            errno: None,
        }
    }
}

pub enum CallName {
    Func(&'static str),
    Ioctl(&'static str),
}

use CallName::Ioctl;

impl From<&'static str> for CallName {
    fn from(func: &'static str) -> Self {
        Self::Func(func)
    }
}

// TODO: Need to make sure signal/exit handlers and signal masks are cleaned up
// when this function returns.

pub fn run<Arg, Fh>(
    args: impl IntoIterator<Item = Arg>,
    filter: &mut Fh,
) -> Result<i32, Error>
where
    Arg: Into<OsString>,
    Fh: FilterHooks,
{
    if !isatty(0).unwrap_or(false) {
        return Err(Error::with_kind(NotATty));
    }

    install_exit_handler()?;
    let term_attrs = tcgetattr(0).map_err(GetAttrFailed.with("tcgetattr"))?;
    let winsize =
        tiocgwinsz(0).map_err(GetSizeFailed.with(Ioctl("TIOCGWINSZ")))?;
    let pty = posix_openpt(OFlag::O_RDWR | OFlag::O_NOCTTY)
        .map_err(CreatePtyFailed.with("posix_openpt"))?;

    grantpt(&pty).map_err(CreatePtyFailed.with("grantpt"))?;
    unlockpt(&pty).map_err(CreatePtyFailed.with("unlockpt"))?;
    let child_tty_name =
        unsafe { ptsname(&pty) }.map_err(CreatePtyFailed.with("ptsname"))?;

    let mut new_attrs = term_attrs.clone();
    cfmakeraw(&mut new_attrs);
    tcsetattr(0, SetArg::TCSANOW, &new_attrs).map_err(
        SetAttrFailed {
            target: Parent,
            caller: Parent,
        }
        .with("tcsetattr"),
    )?;

    ORIG_TERM_ATTRS.with(|attrs| {
        attrs.set(Some(term_attrs.clone()));
    });

    install_terminate_handler()?;
    let child_pid = match {
        unsafe { fork() }.map_err(CreateChildFailed.with("fork"))?
    } {
        ForkResult::Child => {
            drop(pty);
            child_exec(args, child_tty_name.as_str(), &term_attrs, &winsize)?;
            unreachable!();
        }
        ForkResult::Parent {
            child,
        } => child,
    };

    CHILD_PID.with(|pid| {
        pid.set(Some(child_pid));
    });

    let mut orig_sigmask = SigSet::empty();
    sigprocmask(
        SigmaskHow::SIG_BLOCK,
        Some(&{
            let mut set = SigSet::empty();
            set.add(Signal::SIGWINCH);
            set
        }),
        Some(&mut orig_sigmask),
    )
    .map_err(SignalSetupFailed.with("sigprocmask"))?;

    unsafe {
        sigaction(
            Signal::SIGWINCH,
            &SigAction::new(
                SigHandler::Handler(handle_sigwinch),
                SaFlags::SA_RESTART,
                SigSet::empty(),
            ),
        )
    }
    .map_err(SignalSetupFailed.with("sigaction"))?;

    let pty_fd = pty.as_raw_fd();
    // As of nix v0.23.0, `FdSet::insert` and `FdSet::remove` cause UB if
    // passed a file descriptor greater than or equal to `libc::FD_SETSIZE`.
    // As a workaround, we'll check `pty`'s file descriptor manually:
    assert!(usize::try_from(pty_fd).unwrap() < libc::FD_SETSIZE);

    let mut bufs = Buffers {
        input: [0_u8; 1024],
        output: Vec::with_capacity(1024),
    };

    loop {
        let mut fds = FdSet::new();
        fds.insert(0);
        fds.insert(pty_fd);

        match pselect(pty_fd + 1, &mut fds, None, None, None, &orig_sigmask) {
            Err(Errno::EINTR) => {
                update_child_winsize(&pty)?;
                continue;
            }
            r => {
                r.map_err(ChildCommFailed.with("pselect"))?;
            }
        }

        if fds.contains(0) {
            if let ControlFlow::Break(_) =
                handle_stdin_ready(&pty, filter, &mut bufs)?
            {
                break;
            }
        }

        if fds.contains(pty_fd) {
            if let ControlFlow::Break(_) =
                handle_pty_ready(&pty, filter, &mut bufs)?
            {
                break;
            }
        }
    }

    let status = waitpid(child_pid, None)
        .map_err(GetChildStatusFailed.with("waitpid"))?;
    if let WaitStatus::Exited(_, code) = status {
        Ok(code)
    } else {
        Err(Error {
            kind: ErrorKind::UnexpectedChildStatus(UnexpectedChildStatus(
                status,
            )),
            call: Some("waitpid".into()),
            errno: None,
        })
    }
}

pub trait FilterHooks {
    fn on_child_data<F>(&mut self, data: &[u8], mut parent_write: F)
    where
        F: FnMut(&[u8]),
    {
        parent_write(data);
    }

    fn on_parent_data<F>(&mut self, data: &[u8], mut child_write: F)
    where
        F: FnMut(&[u8]),
    {
        child_write(data);
    }
}

pub struct DefaultFilterHooks;

impl FilterHooks for DefaultFilterHooks {}

/// For use with [`chunked`]; see its documentation for more information.
struct Chunked<'a, T, OutFn> {
    buf: &'a mut Vec<T>,
    chunk_out: OutFn,
}

impl<'a, T, OutFn> Chunked<'a, T, OutFn>
where
    T: Copy,
    OutFn: FnMut(&[T]),
{
    /// For use with [`chunked`]; see its documentation for more information.
    pub fn add(&mut self, mut data: &[T]) {
        assert!(self.buf.capacity() > 0);
        loop {
            let spare = self.buf.capacity() - self.buf.len();
            if data.len() <= spare {
                self.buf.extend(data.iter().copied());
                if self.buf.capacity() == self.buf.len() {
                    self.flush();
                }
                return;
            }
            self.buf.extend(data.iter().copied().take(spare));
            self.flush();
            data = &data[spare..];
        }
    }

    fn flush(&mut self) {
        let mut buf = core::mem::take(self.buf);
        (self.chunk_out)(&buf);
        buf.clear();
        *self.buf = buf;
    }
}

/// Given a function, `data_in`, that repeatedly calls [`Chunked::add`] with
/// pieces of data, this function groups the data into chunks using the
/// provided buffer `buf` and calls `chunk_out` with each chunk.
fn chunked<T, In, Out>(buf: &mut Vec<T>, data_in: In, chunk_out: Out)
where
    T: Copy,
    In: FnOnce(&mut Chunked<'_, T, Out>),
    Out: FnMut(&[T]),
{
    let mut chunked = Chunked {
        buf,
        chunk_out,
    };
    data_in(&mut chunked);
    if !chunked.buf.is_empty() {
        chunked.flush();
    }
}
