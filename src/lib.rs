use std::cell::Cell;
use std::convert::TryFrom;
use std::ffi::{CString, OsString};
use std::fmt::Display;
use std::mem::MaybeUninit;
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

fn expect<T, E: Display>(r: Result<T, E>, msg: impl Display) -> T {
    match r {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}: {}", msg, e);
            if cfg!(feature = "panic") {
                panic!("{}: {}", msg, e);
            } else {
                exit(1);
            }
        }
    }
}

fn get_winsize(fd: RawFd) -> nix::Result<libc::winsize> {
    nix::ioctl_read_bad!(tiocgwinsz, libc::TIOCGWINSZ, libc::winsize);
    let mut size = MaybeUninit::uninit();
    unsafe { tiocgwinsz(fd, size.as_mut_ptr()) }?;
    Ok(unsafe { size.assume_init() })
}

fn set_winsize(fd: RawFd, size: &libc::winsize) -> nix::Result<()> {
    nix::ioctl_write_ptr_bad!(tiocswinsz, libc::TIOCSWINSZ, libc::winsize);
    unsafe { tiocswinsz(fd, size as *const _) }.map(|_| ())
}

thread_local! {
    static ORIG_TERM_ATTRS: Cell<Option<Termios>> = Cell::new(None);
    static CHILD_PID: Cell<Option<Pid>> = Cell::new(None);
}

fn install_exit_handler() {
    extern "C" fn handle_exit() {
        ORIG_TERM_ATTRS.with(|attrs| {
            if let Some(attrs) = attrs.take() {
                let _ = tcsetattr(0, SetArg::TCSANOW, &attrs);
            }
        });
    }

    if unsafe { libc::atexit(handle_exit) } != 0 {
        eprintln!("error: could not register exit handler");
        exit(1);
    }
}

fn install_terminate_handler() {
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
        expect(unsafe { sigaction(signal, &action) }, "sigaction()");
    }
}

fn child_exec<Arg, Path>(
    args: impl IntoIterator<Item = Arg>,
    tty_name: &Path,
    attrs: &Termios,
    winsize: &libc::winsize,
) -> !
where
    Arg: Into<OsString>,
    Path: NixPath + ?Sized,
{
    expect(setsid(), "setsid()");
    for fd in 0..=2 {
        let _ = close(fd);
    }

    let tty_fd = expect(
        open(tty_name, OFlag::O_RDWR, Mode::empty()),
        "could not open tty",
    );

    for fd in 0..=2 {
        let _ = dup2(tty_fd, fd);
    }

    if tty_fd > 2 {
        let _ = close(tty_fd);
    }

    expect(
        tcsetattr(0, SetArg::TCSANOW, attrs),
        "could not set child tty attrs",
    );
    expect(set_winsize(0, winsize), "could not set child tty size");

    let args: Vec<_> = args
        .into_iter()
        .map(|a| CString::new(a.into().into_vec()).unwrap())
        .collect();
    expect(execvp(&args[0], &args), "execvp()");
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
) {
    let nread = match read(0, &mut bufs.input) {
        Ok(0) => {
            eprintln!("error: unexpected empty read from parent terminal");
            exit(1);
        }
        r => expect(r, "could not read from parent terminal"),
    };

    let inbuf = &bufs.input;
    chunked(
        &mut bufs.output,
        |c| filter.on_parent_data(&inbuf[..nread], |data| c.add(data)),
        |chunk| {
            if write(pty.as_raw_fd(), chunk).is_err() {
                exit(0);
            }
        },
    );
}

fn handle_pty_ready<Fh: FilterHooks, const N: usize>(
    pty: &PtyMaster,
    filter: &mut Fh,
    bufs: &mut Buffers<N>,
) {
    let nread = match read(pty.as_raw_fd(), &mut bufs.input) {
        Ok(0) | Err(_) => {
            exit(CHILD_PID.with(|pid| match pid.get() {
                Some(pid) => {
                    let status = waitpid(pid, None);
                    let status = expect(status, "waitpid()");
                    if let WaitStatus::Exited(_, code) = status {
                        code
                    } else {
                        0
                    }
                }
                None => 0,
            }));
        }
        Ok(n) => n,
    };

    let inbuf = &bufs.input;
    chunked(
        &mut bufs.output,
        |c| filter.on_child_data(&inbuf[..nread], |data| c.add(data)),
        |chunk| {
            expect(write(0, chunk), "could not write to parent terminal");
        },
    );
}

static SIGWINCH_RECEIVED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigwinch(_: c_int) {
    SIGWINCH_RECEIVED.store(true, Ordering::Relaxed);
}

fn update_child_winsize(pty: &PtyMaster) {
    if !SIGWINCH_RECEIVED.swap(false, Ordering::Relaxed) {
        return;
    }
    let size = expect(get_winsize(0), "could not get terminal size");
    expect(
        set_winsize(pty.as_raw_fd(), &size),
        "could not set child tty size",
    );
}

pub fn run<Arg, Fh>(args: impl IntoIterator<Item = Arg>, filter: &mut Fh) -> !
where
    Arg: Into<OsString>,
    Fh: FilterHooks,
{
    if !isatty(0).unwrap_or(false) {
        eprintln!("error: stdin is not a tty");
        exit(1);
    }

    install_exit_handler();
    let term_attrs = expect(tcgetattr(0), "tcgetattr(0)");
    let winsize = expect(get_winsize(0), "could not get terminal size");
    let pty = expect(
        posix_openpt(OFlag::O_RDWR | OFlag::O_NOCTTY),
        "posix_openpt()",
    );

    expect(grantpt(&pty), "grantpt()");
    expect(unlockpt(&pty), "unlockpt()");
    let child_tty_name = expect(unsafe { ptsname(&pty) }, "ptsname()");

    let mut new_attrs = term_attrs.clone();
    cfmakeraw(&mut new_attrs);
    expect(tcsetattr(0, SetArg::TCSANOW, &new_attrs), "tcsetattr(0)");

    ORIG_TERM_ATTRS.with(|attrs| {
        attrs.set(Some(term_attrs.clone()));
    });

    install_terminate_handler();
    match expect(unsafe { fork() }, "fork()") {
        ForkResult::Child => {
            drop(pty);
            child_exec(args, child_tty_name.as_str(), &term_attrs, &winsize);
        }
        ForkResult::Parent {
            child,
        } => {
            CHILD_PID.with(|pid| {
                pid.set(Some(child));
            });
        }
    }

    let mut orig_sigmask = SigSet::empty();
    expect(
        sigprocmask(
            SigmaskHow::SIG_BLOCK,
            Some(&{
                let mut set = SigSet::empty();
                set.add(Signal::SIGWINCH);
                set
            }),
            Some(&mut orig_sigmask),
        ),
        "sigprocmask()",
    );

    expect(
        unsafe {
            sigaction(
                Signal::SIGWINCH,
                &SigAction::new(
                    SigHandler::Handler(handle_sigwinch),
                    SaFlags::SA_RESTART,
                    SigSet::empty(),
                ),
            )
        },
        "sigaction()",
    );

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
                update_child_winsize(&pty);
                continue;
            }
            r => {
                expect(r, "pselect()");
            }
        }

        if fds.contains(0) {
            handle_stdin_ready(&pty, filter, &mut bufs);
        }

        if fds.contains(pty_fd) {
            handle_pty_ready(&pty, filter, &mut bufs);
        }
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
