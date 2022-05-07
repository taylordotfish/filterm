/*
 * This file contains code originally from version 2.30 of the the GNU
 * C Library (specifically, the file termios/cfmakeraw.c). That code is
 * covered by the following copyright and license notice:
 *
 *     Copyright (C) 1992-2019 Free Software Foundation, Inc.
 *     This file is part of the GNU C Library.
 *
 *     The GNU C Library is free software; you can redistribute it and/or
 *     modify it under the terms of the GNU Lesser General Public
 *     License as published by the Free Software Foundation; either
 *     version 2.1 of the License, or (at your option) any later version.
 *
 *     The GNU C Library is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *     Lesser General Public License for more details.
 *
 *     You should have received a copy of the GNU Lesser General Public
 *     License along with the GNU C Library; if not, see
 *     <http://www.gnu.org/licenses/>.
 *
 * All modifications in this file relative to the original GNU C Library
 * code are released under the same license as that code.
 */

use nix::sys::termios::{ControlFlags, InputFlags, LocalFlags, OutputFlags};
use nix::sys::termios::{SpecialCharacterIndices, Termios};

pub fn cfmakeraw(t: &mut Termios) {
    t.input_flags &= !(InputFlags::IGNBRK
        | InputFlags::BRKINT
        | InputFlags::PARMRK
        | InputFlags::ISTRIP
        | InputFlags::INLCR
        | InputFlags::IGNCR
        | InputFlags::ICRNL
        | InputFlags::IXON);
    t.output_flags &= !OutputFlags::OPOST;
    t.local_flags &= !(LocalFlags::ECHO
        | LocalFlags::ECHONL
        | LocalFlags::ICANON
        | LocalFlags::ISIG
        | LocalFlags::IEXTEN);
    t.control_flags &= !(ControlFlags::CSIZE | ControlFlags::PARENB);
    t.control_flags |= ControlFlags::CS8;
    // read returns when one char is available.
    t.control_chars[SpecialCharacterIndices::VMIN as usize] = 1;
    t.control_chars[SpecialCharacterIndices::VTIME as usize] = 0;
}
