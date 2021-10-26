/*
 * Copyright (C) 2021 taylor.fish <contact@taylor.fish>
 *
 * This file is part of Filterm.
 *
 * Filterm is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Filterm is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Filterm. If not, see <https://www.gnu.org/licenses/>.
 */

//! Filterm lets you run a child process while piping all terminal data to
//! and from the child through a custom filter. This lets you modify things
//! like ANSI escape sequences that get sent from the child.
//!
//! The main way of using Filterm is to define a custom filter by implementing
//! the [`FilterHooks`] trait, and then call [`run`].
//!
//! For an example of Filterm in use, see
//! [Monoterm](https://github.com/taylordotfish/monoterm).
//!
//! Platform support
//! ----------------
//!
//! Filterm has been tested on GNU/Linux. It may work on other Unix-like
//! operating systems, as it avoids using Linux- and GNU-specific
//! functionality.

use std::cell::Cell;
use std::convert::{Infallible, TryFrom};
use std::ffi::{CString, OsString};
use std::mem::MaybeUninit;
use std::ops::ControlFlow;
use std::os::raw::c_int;
use std::os::unix::ffi::OsStringExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

pub use nix;
use nix::errno::Errno;
use nix::fcntl::{fcntl, open, FcntlArg, FdFlag, OFlag};
use nix::libc;
use nix::pty::{grantpt, posix_openpt, ptsname, unlockpt, PtyMaster};
use nix::sys::select::{pselect, FdSet};
use nix::sys::signal::{kill, raise, sigprocmask, SigSet, SigmaskHow, Signal};
use nix::sys::signal::{sigaction, SaFlags, SigAction, SigHandler};
use nix::sys::stat::Mode;
use nix::sys::termios::{tcgetattr, tcsetattr, SetArg, Termios};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::{close, dup2, execvp, fork, isatty, read, setsid, write};
use nix::unistd::{pipe, ForkResult, Pid};
use nix::NixPath;

mod cfmakeraw;
#[macro_use]
pub mod error;
#[allow(dead_code)]
mod owned_fd;

use cfmakeraw::cfmakeraw;
pub use error::Error;
use error::{CallName::Ioctl, Client::*, ErrorKind::*};
use owned_fd::OwnedFd;

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
    static ORIG_TERM_ATTRS: Cell<Option<Termios>> = Cell::default();
    static CHILD_PID: Cell<Option<Pid>> = Cell::default();
    static ORIG_TERMINATE_ACTIONS: [Cell<Option<SigAction>>; 3] =
        Default::default();
    static ORIG_SIGWINCH_ACTION: Cell<Option<SigAction>> = Cell::default();
    static ORIG_SIGCHLD_ACTION: Cell<Option<SigAction>> = Cell::default();
    static ORIG_SIGMASK: Cell<Option<SigSet>> = Cell::default();
}

const TERMINATE_SIGNALS: [Signal; 3] =
    [Signal::SIGHUP, Signal::SIGINT, Signal::SIGTERM];

static PENDING_TERMINATE: AtomicI32 = AtomicI32::new(i32::MIN);

fn install_terminate_handler() -> Result<(), Error> {
    extern "C" fn handle_terminate(signal: c_int) {
        #[allow(clippy::useless_conversion)]
        let signal = i32::try_from(signal).unwrap();
        assert!(signal != i32::MIN);
        PENDING_TERMINATE.store(signal, Ordering::Relaxed);
    }

    let action = SigAction::new(
        SigHandler::Handler(handle_terminate),
        SaFlags::empty(),
        SigSet::empty(),
    );

    ORIG_TERMINATE_ACTIONS.with(|actions| {
        IntoIterator::into_iter(TERMINATE_SIGNALS).zip(actions).try_for_each(
            |(signal, orig)| {
                unsafe { sigaction(signal, &action) }
                    .map_err(SignalSetupFailed.with("sigaction"))
                    .map(|a| orig.set(Some(a)))
            },
        )
    })
}

fn handle_pending_terminate() -> Result<(), Error> {
    match PENDING_TERMINATE.swap(i32::MIN, Ordering::Relaxed) {
        i32::MIN => Ok(()),
        signal => Err(Error::with_kind(ReceivedSignal(
            Signal::try_from(signal).expect("invalid signal"),
        ))),
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Eq, PartialEq)]
enum ChildErrorKind {
    Setsid = 1,
    Sigprocmask = 2,
    OpenTty = 3,
    Tcsetattr = 4,
    Tiocswinsz = 5,
    Execvp = 6,
}

struct ChildError(ChildErrorKind, Errno);

impl From<ChildError> for Error {
    fn from(e: ChildError) -> Self {
        use ChildErrorKind as Kind;
        Self {
            kind: match e.0 {
                Kind::Setsid => ChildSetupFailed,
                Kind::Sigprocmask => ChildSetupFailed,
                Kind::OpenTty => ChildOpenTtyFailed,
                Kind::Tcsetattr => SetAttrFailed {
                    target: Child,
                    caller: Child,
                },
                Kind::Tiocswinsz => SetSizeFailed {
                    target: Child,
                    caller: Child,
                },
                Kind::Execvp => ChildExecFailed,
            },
            call: match e.0 {
                Kind::Setsid => Some("setsid".into()),
                Kind::Sigprocmask => Some("sigprocmask".into()),
                Kind::OpenTty => Some("open".into()),
                Kind::Tcsetattr => Some("tcsetattr".into()),
                Kind::Tiocswinsz => Some(Ioctl("TIOCSWINSZ")),
                Kind::Execvp => Some("execvp".into()),
            },
            errno: Some(e.1),
        }
    }
}

impl From<ChildError> for u64 {
    fn from(e: ChildError) -> Self {
        ((e.0 as u32 as u64) << 32) | (e.1 as i32 as u32 as u64)
    }
}

impl TryFrom<u64> for ChildError {
    type Error = u64;

    fn try_from(n: u64) -> Result<Self, Self::Error> {
        use ChildErrorKind::*;
        let errno = Errno::from_i32(n as u32 as i32);
        let kind = match (n >> 32) as u32 {
            1 => Setsid,
            2 => Sigprocmask,
            3 => OpenTty,
            4 => Tcsetattr,
            5 => Tiocswinsz,
            6 => Execvp,
            _ => return Err(n),
        };
        Ok(Self(kind, errno))
    }
}

fn child_exec<Arg, Path>(
    args: impl IntoIterator<Item = Arg>,
    tty_name: &Path,
    attrs: &Termios,
    winsize: &libc::winsize,
) -> Result<Infallible, ChildError>
where
    Arg: Into<OsString>,
    Path: NixPath + ?Sized,
{
    use ChildError as Error;
    use ChildErrorKind::*;

    sigprocmask(
        SigmaskHow::SIG_SETMASK,
        Some(&ORIG_SIGMASK.with(|mask| mask.take()).unwrap()),
        None,
    )
    .map_err(|e| Error(Sigprocmask, e))?;

    setsid().map_err(|e| Error(Setsid, e))?;
    for fd in 0..=2 {
        let _ = close(fd);
    }

    let tty_fd = open(tty_name, OFlag::O_RDWR, Mode::empty())
        .map_err(|e| Error(OpenTty, e))?;

    for fd in 0..=2 {
        let _ = dup2(tty_fd, fd);
    }

    if tty_fd > 2 {
        let _ = close(tty_fd);
    }

    tcsetattr(0, SetArg::TCSANOW, attrs).map_err(|e| Error(Tcsetattr, e))?;
    tiocswinsz(0, winsize).map_err(|e| Error(Tiocswinsz, e))?;

    let args: Vec<_> = args
        .into_iter()
        .map(|a| CString::new(a.into().into_vec()).unwrap())
        .collect();
    execvp(&args[0], &args).map_err(|e| Error(Execvp, e))?;
    unreachable!();
}

fn read_child_error(fd: RawFd) -> Result<Option<ChildError>, Error> {
    let mut buf = [0; 8];
    let mut nread = 0;
    while nread < buf.len() {
        nread += match read(fd, &mut buf[nread..])
            .map_err(ChildCommFailed.with("read"))?
        {
            0 => return Ok(None),
            n => n,
        }
    }
    let n = u64::from_be_bytes(buf);
    Ok(Some(
        ChildError::try_from(n)
            .expect("received invalid error from child process"),
    ))
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
            if !write_err {
                write_err = write(pty.as_raw_fd(), chunk).is_err();
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

static SIGCHLD_RECEIVED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigchld(_: c_int) {
    SIGCHLD_RECEIVED.store(true, Ordering::Relaxed);
}

fn try_child_wait(pid: Pid) -> Result<Option<Exit>, Error> {
    if !SIGCHLD_RECEIVED.swap(false, Ordering::Relaxed) {
        return Ok(None);
    }
    match waitpid(pid, Some(WaitPidFlag::WNOHANG))
        .map_err(GetChildStatusFailed.with("waitpid"))?
    {
        WaitStatus::Exited(_, code) => Ok(Some(Exit::Normal(code))),
        WaitStatus::Signaled(_, sig, _) => Ok(Some(Exit::Signal(sig))),
        WaitStatus::StillAlive => Ok(None),
        status => Err(Error {
            kind: UnexpectedChildStatus(status),
            call: Some("waitpid".into()),
            errno: None,
        }),
    }
}

/// A trait for filtering data to and from a child terminal.
///
/// An object implementing this trait should be passed to [`run`].
pub trait FilterHooks {
    /// Called when data from the child terminal is received.
    ///
    /// `parent_write` should be called (any number of times) to forward this
    /// data, or send different data, to the parent terminal. The default
    /// implementation of this method forwards all data unchanged.
    fn on_child_data<F>(&mut self, data: &[u8], mut parent_write: F)
    where
        F: FnMut(&[u8]),
    {
        parent_write(data);
    }

    /// Called when data from the parent terminal is received.
    ///
    /// `child_write` should be called (any number of times) to forward this
    /// data, or send different data, to the child terminal. The default
    /// implementation of this method forwards all data unchanged.
    fn on_parent_data<F>(&mut self, data: &[u8], mut child_write: F)
    where
        F: FnMut(&[u8]),
    {
        child_write(data);
    }
}

/// An implementation of [`FilterHooks`] that simply uses the default methods.
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

/// Returned by [`run`] when the child process exits.
#[non_exhaustive]
pub enum Exit {
    Normal(i32),
    Signal(Signal),
}

fn run_impl<Args, Fh>(args: Args, filter: &mut Fh) -> Result<Exit, Error>
where
    Args: IntoIterator<Item = OsString>,
    Fh: FilterHooks,
{
    if !isatty(0).unwrap_or(false) {
        return Err(Error::with_kind(NotATty));
    }

    let term_attrs = tcgetattr(0).map_err(GetAttrFailed.with("tcgetattr"))?;
    let winsize =
        tiocgwinsz(0).map_err(GetSizeFailed.with(Ioctl("TIOCGWINSZ")))?;
    let pty = posix_openpt(OFlag::O_RDWR | OFlag::O_NOCTTY)
        .map_err(CreatePtyFailed.with("posix_openpt"))?;

    let pty_fd = pty.as_raw_fd();
    // As of nix v0.23.0, `FdSet::insert` and `FdSet::remove` cause UB if
    // passed a file descriptor greater than or equal to `libc::FD_SETSIZE`,
    // so it's important we check the file descriptor manually.
    if !usize::try_from(pty_fd).map_or(false, |fd| fd < libc::FD_SETSIZE) {
        return Err(Error::with_kind(BadPtyFd(pty_fd)));
    }

    grantpt(&pty).map_err(CreatePtyFailed.with("grantpt"))?;
    unlockpt(&pty).map_err(CreatePtyFailed.with("unlockpt"))?;
    let child_tty_name =
        unsafe { ptsname(&pty) }.map_err(CreatePtyFailed.with("ptsname"))?;

    let mut orig_sigmask = SigSet::empty();
    sigprocmask(
        SigmaskHow::SIG_BLOCK,
        Some(&{
            let mut set = SigSet::empty();
            IntoIterator::into_iter([Signal::SIGWINCH, Signal::SIGCHLD])
                .chain(TERMINATE_SIGNALS)
                .for_each(|s| set.add(s));
            set
        }),
        Some(&mut orig_sigmask),
    )
    .map_err(SignalSetupFailed.with("sigprocmask"))?;
    ORIG_SIGMASK.with(|mask| mask.set(Some(orig_sigmask)));
    install_terminate_handler()?;

    let mut new_attrs = term_attrs.clone();
    cfmakeraw(&mut new_attrs);
    tcsetattr(0, SetArg::TCSANOW, &new_attrs).map_err(
        SetAttrFailed {
            target: Parent,
            caller: Parent,
        }
        .with("tcsetattr"),
    )?;
    ORIG_TERM_ATTRS.with(|attrs| attrs.set(Some(term_attrs.clone())));

    let (read_fd, write_fd) = pipe().map_err(ChildCommFailed.with("pipe"))?;
    let [read_fd, write_fd] = [read_fd, write_fd].map(OwnedFd::new);
    let fork = unsafe { fork() }.map_err(CreateChildFailed.with("fork"))?;
    let child_pid = match fork {
        ForkResult::Child => {
            drop(pty);
            ignore_error!(read_fd.close());
            ignore_error!(fcntl(
                write_fd.get(),
                FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC)
            ));
            let result = child_exec(
                args,
                child_tty_name.as_str(),
                &term_attrs,
                &winsize,
            );
            let err = u64::from(result.unwrap_err()).to_be_bytes();
            ignore_error!(write(write_fd.get(), &err));
            unsafe { libc::_exit(1) };
        }
        ForkResult::Parent {
            child,
        } => child,
    };

    CHILD_PID.with(|pid| pid.set(Some(child_pid)));
    ignore_error!(write_fd.close());
    if let Some(e) = read_child_error(read_fd.get())? {
        return Err(Error::from(e));
    }
    ignore_error!(read_fd.close());

    let orig_sigwinch = unsafe {
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
    ORIG_SIGWINCH_ACTION.with(|act| act.set(Some(orig_sigwinch)));

    let orig_sigchld = unsafe {
        sigaction(
            Signal::SIGCHLD,
            &SigAction::new(
                SigHandler::Handler(handle_sigchld),
                SaFlags::SA_RESTART,
                SigSet::empty(),
            ),
        )
    }
    .map_err(SignalSetupFailed.with("sigaction"))?;
    ORIG_SIGCHLD_ACTION.with(|act| act.set(Some(orig_sigchld)));

    const BUFFER_SIZE: usize = 1024;
    let mut bufs = Buffers {
        input: [0; BUFFER_SIZE],
        output: Vec::with_capacity(BUFFER_SIZE),
    };

    loop {
        let mut fds = FdSet::new();
        fds.insert(0);
        fds.insert(pty_fd);

        match pselect(pty_fd + 1, &mut fds, None, None, None, &orig_sigmask) {
            Err(Errno::EINTR) => {
                handle_pending_terminate()?;
                if let Some(exit) = try_child_wait(child_pid)? {
                    return Ok(exit);
                }
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

    match waitpid(child_pid, None)
        .map_err(GetChildStatusFailed.with("waitpid"))?
    {
        WaitStatus::Exited(_, code) => Ok(Exit::Normal(code)),
        WaitStatus::Signaled(_, sig, _) => Ok(Exit::Signal(sig)),
        status => Err(Error {
            kind: UnexpectedChildStatus(status),
            call: Some("waitpid".into()),
            errno: None,
        }),
    }
}

/// Runs the command specified by `args` with the given filter.
///
/// The arguments in `args` are passed to `execvp()`.
pub fn run<Args, Arg, Fh>(args: Args, filter: &mut Fh) -> Result<Exit, Error>
where
    Args: IntoIterator<Item = Arg>,
    Arg: Into<OsString>,
    Fh: FilterHooks,
{
    static HAS_RUN: AtomicBool = AtomicBool::new(false);
    thread_local! {
        static HAS_RUN_ON_THIS_THREAD: Cell<bool> = Cell::new(false);
    }

    if !HAS_RUN_ON_THIS_THREAD.with(|b| b.get()) {
        if HAS_RUN.swap(true, Ordering::Relaxed) {
            panic!("`run` may not be called from multiple threads");
        }
        HAS_RUN_ON_THIS_THREAD.with(|b| b.set(true));
    }

    let result = run_impl(args.into_iter().map(|a| a.into()), filter);
    if let Some(pid) = CHILD_PID.with(|pid| pid.take()) {
        if result.is_err() {
            ignore_error!(kill(pid, Signal::SIGHUP));
        }
    }
    if let Some(attrs) = ORIG_TERM_ATTRS.with(|attrs| attrs.take()) {
        ignore_error!(tcsetattr(0, SetArg::TCSANOW, &attrs));
    }
    if let Some(action) = ORIG_SIGWINCH_ACTION.with(|a| a.take()) {
        ignore_error!(unsafe { sigaction(Signal::SIGWINCH, &action) });
    }
    if let Some(action) = ORIG_SIGCHLD_ACTION.with(|a| a.take()) {
        ignore_error!(unsafe { sigaction(Signal::SIGCHLD, &action) });
    }
    ORIG_TERMINATE_ACTIONS.with(|actions| {
        actions
            .iter()
            .zip(TERMINATE_SIGNALS)
            .filter_map(|(a, s)| a.take().map(|a| (a, s)))
            .for_each(|(action, signal)| {
                ignore_error!(unsafe { sigaction(signal, &action) });
            });
    });
    if let Some(mask) = ORIG_SIGMASK.with(|mask| mask.take()) {
        ignore_error!(sigprocmask(SigmaskHow::SIG_SETMASK, Some(&mask), None));
    }
    if let Err(e) = handle_pending_terminate().as_ref().and(result.as_ref()) {
        if let ReceivedSignal(signal) = e.kind {
            ignore_error!(raise(signal));
        }
    }
    result
}
