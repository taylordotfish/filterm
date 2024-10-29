/*
 * Copyright (C) 2021-2022, 2024 taylor.fish <contact@taylor.fish>
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

//! Error types.

use nix::errno::Errno;
use nix::poll::PollFlags;
use nix::sys::wait::WaitStatus;
use std::fmt::{self, Display, Formatter};
use std::io;

/// Indicates whether an error occurred in the parent or child
/// process/terminal.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Client {
    /// The parent process/terminal.
    Parent,
    /// The child process/terminal.
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

/// Error information for [`ErrorKind::ChildCommFailed`].
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

/// Error information for [`ErrorKind::UnexpectedChildStatus`].
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

/// The kind of error that occurred.
#[non_exhaustive]
#[derive(Debug)]
pub enum ErrorKind {
    /// Stdin is not a TTY.
    #[non_exhaustive]
    NotATty,
    /// Could not get terminal attributes.
    #[non_exhaustive]
    GetAttrFailed,
    /// Could not set terminal attributes.
    #[non_exhaustive]
    SetAttrFailed {
        /// The client that tried to set terminal attributes.
        target: Client,
        /// The client whose terminal attributes were attempted to be set.
        caller: Client,
    },
    /// Could not get terminal size.
    #[non_exhaustive]
    GetSizeFailed,
    /// Could not set terminal size.
    #[non_exhaustive]
    SetSizeFailed,
    /// Could not create a pseudoterminal.
    #[non_exhaustive]
    CreatePtyFailed,
    /// Could not create the child process.
    #[non_exhaustive]
    CreateChildFailed,
    /// Could not set up signal handlers.
    #[non_exhaustive]
    SignalSetupFailed,
    /// Could not communicate with the child process.
    #[non_exhaustive]
    ChildCommFailed(ChildCommFailed),
    /// Could not set up the child process.
    #[non_exhaustive]
    ChildSetupFailed,
    /// Could not open the terminal in the child process.
    #[non_exhaustive]
    ChildOpenTtyFailed,
    /// Could not execute the requested process.
    #[non_exhaustive]
    ChildExecFailed,
    /// Received no data when trying to read from the parent terminal.
    #[non_exhaustive]
    EmptyParentRead,
    /// Could not read from the parent terminal.
    #[non_exhaustive]
    ParentReadFailed,
    /// Could not write to the parent terminal.
    #[non_exhaustive]
    ParentWriteFailed,
    /// Could not get the status of the child process.
    #[non_exhaustive]
    GetChildStatusFailed,
    /// Received an unexpected status from the child process.
    #[non_exhaustive]
    UnexpectedChildStatus(UnexpectedChildStatus),
    /// Parent process received a termination signal.
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

/// The name of the function or `ioctl()` request that produced an error.
#[non_exhaustive]
#[derive(Debug)]
pub enum CallName {
    /// The name of the function that produced an error.
    #[non_exhaustive]
    Func(&'static str),
    /// The name of the `ioctl()` request that produced an error.
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
    /// The kind of error that occurred.
    pub kind: ErrorKind,
    /// The name of the function or `ioctl()` request (if any) that produced
    /// an error.
    pub call: Option<CallName>,
    /// The underlying system error (if any) that occurred.
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
            .map(Errno::from_raw);
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
        if let ::std::result::Result::Err(ref e) = $expr {
            if ::std::cfg!(debug_assertions) {
                ::std::eprintln!(
                    "{}:{}: {:?}",
                    ::std::file!(),
                    ::std::line!(),
                    e,
                );
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
    ($expr:expr) => {{
        let result = $expr;
        if !::std::cfg!(debug_assertions) {
        } else if let ::std::result::Result::Err(ref e) = result {
            use ::nix::unistd::write;
            use ::std::os::unix::io::AsFd;
            use $crate::error::ErrorWrapper;
            #[allow(unused_imports)]
            use $crate::error::IntoRawErrno;
            use $crate::utils::format_u32;

            let stderr = ::std::io::stderr();
            let stderr = AsFd::as_fd(&stderr);
            let _ = write(stderr, b"error at ");
            let _ = write(stderr, file!().as_bytes());
            let _ = write(stderr, b":");

            const LINE: ([u8; 10], usize) = format_u32(::std::line!());
            let _ = write(stderr, &LINE.0[LINE.1..]);

            if let Some(errno) = ErrorWrapper(e).into_raw_errno() {
                let _ = write(stderr, b": os error ");
                let (buf, i) = format_u32(errno as u32);
                let _ = write(stderr, &buf[i..]);
            }
            let _ = write(stderr, b"\n");
        }
    }};
}

#[allow(dead_code)]
fn allow_unused() {
    // Suppress warnings if `IntoRawErrno` is unused, since it's part of a
    // specialization hack.
    ().into_raw_errno();
}
