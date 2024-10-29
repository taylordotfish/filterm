/*
 * Copyright (C) 2022 taylor.fish <contact@taylor.fish>
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

use nix::libc::c_uint;

macro_rules! define_format_uint {
    ($vis:vis $name:ident, $type:ty) => {
        /// Const function that formats an integer of type
        #[doc = ::core::concat!("[`", ::core::stringify!($type), "`]")]
        /// as a base-10 string. Returns a buffer and the index at which the
        /// formatted number begins.
        $vis const fn $name(
            mut n: $type,
        ) -> (
            [u8; (::core::mem::size_of::<$type>() * 8 * 7 + 22) / 23],
            usize,
        ) {
            let mut buf =
                [b'0'; (::core::mem::size_of::<$type>() * 8 * 7 + 22) / 23];
            let mut i = buf.len() - (n == 0) as usize;
            while n > 0 {
                i -= 1;
                buf[i] = b'0' + (n % 10) as u8;
                n /= 10;
            }
            (buf, i)
        }
    };
}

define_format_uint!(pub format_c_uint, c_uint);
define_format_uint!(pub format_u32, u32);
