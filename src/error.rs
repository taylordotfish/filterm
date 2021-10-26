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

use nix::errno::Errno;
use nix::sys::signal::Signal;
use nix::sys::wait::WaitStatus;
use std::fmt::{self, Display, Formatter};
use std::os::unix::io::RawFd;

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
    SetSizeFailed {
        target: Client,
        caller: Client,
    },
    #[non_exhaustive]
    CreatePtyFailed,
    #[non_exhaustive]
    BadPtyFd(RawFd),
    #[non_exhaustive]
    CreateChildFailed,
    #[non_exhaustive]
    SignalSetupFailed,
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
    #[non_exhaustive]
    UnexpectedChildStatus(WaitStatus),
    #[non_exhaustive]
    ReceivedSignal(Signal),
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
            errno: Some(errno),
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
                write!(f, "could not set {} terminal attributes", target)?;
                if target != caller {
                    write!(f, " from {}", caller)?;
                }
                Ok(())
            }
            GetSizeFailed => write!(f, "could not get terminal size"),
            SetSizeFailed {
                target,
                caller,
            } => {
                write!(f, "could not set {} terminal size", target)?;
                if target != caller {
                    write!(f, " from {}", caller)?;
                }
                Ok(())
            }
            CreatePtyFailed => write!(f, "could not create pseudoterminal"),
            BadPtyFd(fd) => write!(
                f,
                "pseudoterminal file descriptor is too large: {}",
                fd,
            ),
            CreateChildFailed => write!(f, "could not create child process"),
            SignalSetupFailed => write!(f, "could not configure signals"),
            ChildCommFailed => {
                write!(f, "could not communicate with child process")
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
            UnexpectedChildStatus(status) => {
                write!(f, "got unexpected child process status: {:?}", status)
            }
            ReceivedSignal(signal) => {
                write!(f, "process received signal: {}", signal)
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
            Self::Func(name) => write!(f, "{}()", name),
            Self::Ioctl(name) => write!(f, "ioctl {}", name),
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
    pub errno: Option<Errno>,
}

impl Error {
    pub(crate) fn with_kind(kind: impl Into<ErrorKind>) -> Self {
        Self {
            kind: kind.into(),
            call: None,
            errno: None,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        match (&self.call, self.errno) {
            (Some(call), Some(e)) => {
                write!(f, " ({} returned {})", call, e)
            }
            (Some(call), None) => {
                write!(f, " (from {})", call)
            }
            (None, Some(e)) => {
                write!(f, " (got {})", e)
            }
            (None, None) => Ok(()),
        }
    }
}

impl std::error::Error for Error {}

macro_rules! ignore_error {
    ($expr:expr) => {
        match $expr {
            Ok(_) => {}
            Err(e) => {
                if cfg!(debug_assertions) {
                    eprintln!("{}:{}: {:?}", file!(), line!(), e);
                }
            }
        }
    };
}
