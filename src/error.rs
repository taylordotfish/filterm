/*
 * Copyright (C) 2021-2022 taylor.fish <contact@taylor.fish>
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

use nix::errno::Errno;
use nix::poll::PollFlags;
use nix::sys::wait::WaitStatus;
use std::fmt::{self, Display, Formatter};
use std::io;

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Client {
    Parent,
    Child,
}

impl Display for Client {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parent => write!(f, "parent"),
            Self::Child => write!(f, "child"),
        }
    }
}

#[derive(Debug)]
pub struct ChildCommFailed {
    pub(crate) kind: ChildCommFailedKind,
}

impl ChildCommFailed {
    pub(crate) fn new() -> Self {
        ChildCommFailedKind::Unspecified.into()
    }
}

impl From<ChildCommFailed> for ErrorKind {
    fn from(err: ChildCommFailed) -> Self {
        Self::ChildCommFailed(err)
    }
}

#[derive(Debug)]
pub(crate) enum ChildCommFailedKind {
    Unspecified,
    BadPollFlags(PollFlags),
    SignalReadError,
}

impl From<ChildCommFailedKind> for ChildCommFailed {
    fn from(kind: ChildCommFailedKind) -> Self {
        Self {
            kind,
        }
    }
}

#[derive(Debug)]
pub struct UnexpectedChildStatus {
    pub(crate) status: WaitStatus,
}

impl UnexpectedChildStatus {
    pub(crate) fn new(status: WaitStatus) -> Self {
        Self {
            status,
        }
    }
}

impl From<UnexpectedChildStatus> for ErrorKind {
    fn from(err: UnexpectedChildStatus) -> Self {
        Self::UnexpectedChildStatus(err)
    }
}

#[non_exhaustive]
#[derive(Debug)]
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
    SetSizeFailed,
    #[non_exhaustive]
    CreatePtyFailed,
    #[non_exhaustive]
    CreateChildFailed,
    #[non_exhaustive]
    SignalSetupFailed,
    #[non_exhaustive]
    ChildCommFailed(ChildCommFailed),
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
    #[non_exhaustive]
    UnexpectedChildStatus(UnexpectedChildStatus),
    #[non_exhaustive]
    ReceivedSignal(i32),
}

use ErrorKind::*;

impl ErrorKind {
    pub(crate) fn with(
        self,
        name: impl Into<CallName>,
    ) -> impl FnOnce(Errno) -> Error {
        let name = name.into();
        |errno| Error {
            kind: self,
            call: Some(name),
            io_error: Some(errno.into()),
        }
    }
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            NotATty => write!(f, "stdin is not a TTY"),
            GetAttrFailed => write!(f, "could not get terminal attributes"),
            SetAttrFailed {
                target,
                caller,
            } => {
                write!(f, "could not set {target} terminal attributes")?;
                if target != caller {
                    write!(f, " from {caller}")?;
                }
                Ok(())
            }
            GetSizeFailed => write!(f, "could not get terminal size"),
            SetSizeFailed => write!(f, "could not set child terminal size"),
            CreatePtyFailed => write!(f, "could not create pseudoterminal"),
            CreateChildFailed => write!(f, "could not create child process"),
            SignalSetupFailed => write!(f, "could not configure signals"),
            ChildCommFailed(ChildCommFailed {
                kind,
            }) => {
                write!(f, "could not communicate with child process")?;
                match kind {
                    ChildCommFailedKind::Unspecified => {}
                    ChildCommFailedKind::BadPollFlags(flags) => {
                        write!(f, ": bad poll() flags: {flags:?}")?;
                    }
                    ChildCommFailedKind::SignalReadError => {
                        write!(f, ": error reading from signal pipe")?;
                    }
                }
                Ok(())
            }
            ChildSetupFailed => write!(f, "could not set up child process"),
            ChildOpenTtyFailed => {
                write!(f, "could not open terminal in child process")
            }
            ChildExecFailed => write!(f, "could not execute command"),
            EmptyParentRead => {
                write!(f, "unexpected empty read from parent terminal")
            }
            ParentReadFailed => {
                write!(f, "could not read from parent terminal")
            }
            ParentWriteFailed => {
                write!(f, "could not write to parent terminal")
            }
            GetChildStatusFailed => {
                write!(f, "could not get status of child process")
            }
            UnexpectedChildStatus(UnexpectedChildStatus {
                status,
            }) => {
                write!(f, "got unexpected child process status: {status:?}")
            }
            ReceivedSignal(signal) => {
                write!(f, "process received signal: {signal}")
            }
        }
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum CallName {
    #[non_exhaustive]
    Func(&'static str),
    #[non_exhaustive]
    Ioctl(&'static str),
}

impl Display for CallName {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Func(name) => write!(f, "{name}()"),
            Self::Ioctl(name) => write!(f, "ioctl {name}"),
        }
    }
}

impl From<&'static str> for CallName {
    fn from(func: &'static str) -> Self {
        Self::Func(func)
    }
}

/// Returned by [`run`](crate::run) when an error occurs.
#[non_exhaustive]
#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub call: Option<CallName>,
    pub io_error: Option<io::Error>,
}

impl Error {
    pub(crate) fn from_kind(kind: impl Into<ErrorKind>) -> Self {
        Self {
            kind: kind.into(),
            call: None,
            io_error: None,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let errno = self
            .io_error
            .as_ref()
            .and_then(|e| e.raw_os_error())
            .map(Errno::from_i32);
        write!(f, "{}", self.kind)?;
        match (&self.call, errno) {
            (Some(call), Some(e)) => {
                write!(f, " ({call} returned {e})")
            }
            (Some(call), None) => {
                write!(f, " (from {call})")
            }
            (None, Some(e)) => {
                write!(f, " (got {e})")
            }
            (None, None) => Ok(()),
        }
    }
}

impl std::error::Error for Error {}

macro_rules! ignore_error {
    ($expr:expr) => {
        if let Err(ref e) = $expr {
            if cfg!(debug_assertions) {
                eprintln!("{}:{}: {:?}", file!(), line!(), e);
            }
        }
    };
}

pub(crate) struct ErrorWrapper<T>(pub T);

impl ErrorWrapper<&Errno> {
    pub fn into_raw_errno(self) -> Option<i32> {
        Some(*self.0 as _)
    }
}

pub(crate) trait IntoRawErrno: Sized {
    fn into_raw_errno(self) -> Option<i32> {
        None
    }
}

impl<T> IntoRawErrno for T {}

/// Async-signal-safe version of `ignore_error`. Prints only limited error
/// details to maintain signal safety.
macro_rules! ignore_error_sigsafe {
    ($expr:expr) => {
        loop {
            let result = &$expr;
            let e = match result {
                Err(e) if cfg!(debug_assertions) => e,
                _ => break,
            };

            use ::nix::libc::STDERR_FILENO as FD;
            use ::nix::unistd::write;
            use $crate::error::ErrorWrapper;
            #[allow(unused_imports)]
            use $crate::error::IntoRawErrno;
            use $crate::utils::format_u32;

            let _ = write(FD, b"error at ");
            let _ = write(FD, file!().as_bytes());
            let _ = write(FD, b":");

            const LINE: ([u8; 10], usize) = format_u32(line!());
            let _ = write(FD, &LINE.0[LINE.1..]);

            if let Some(errno) = ErrorWrapper(e).into_raw_errno() {
                let _ = write(FD, b": os error ");
                let (buf, i) = format_u32(errno as u32);
                let _ = write(FD, &buf[i..]);
            }
            let _ = write(FD, b"\n");
            break;
        }
    };
}
