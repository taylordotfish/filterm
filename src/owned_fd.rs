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

use nix::unistd::close;
use std::mem::ManuallyDrop;
use std::os::unix::io::RawFd;

pub struct OwnedFd(RawFd);

impl OwnedFd {
    pub fn new(fd: RawFd) -> Self {
        Self(fd)
    }

    pub fn get(&self) -> RawFd {
        self.0
    }

    pub fn close(self) -> nix::Result<()> {
        close(self.release())
    }

    pub fn release(self) -> RawFd {
        ManuallyDrop::new(self).get()
    }
}

impl Drop for OwnedFd {
    fn drop(&mut self) {
        let _ = close(self.get());
    }
}
