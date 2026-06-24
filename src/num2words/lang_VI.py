# -*- coding: utf-8 -*-
# Copyright (c) 2003, Taro Ogawa.  All Rights Reserved.
# Copyright (c) 2013, Savoir-faire Linux inc.  All Rights Reserved.

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301 USA

from __future__ import unicode_literals

import re

to_19 = (u'không', u'một', u'hai', u'ba', u'bốn', u'năm', u'sáu',
         u'bảy', u'tám', u'chín', u'mười', u'mười một', u'mười hai',
         u'mười ba', u'mười bốn', u'mười lăm', u'mười sáu', u'mười bảy',
         u'mười tám', u'mười chín')
tens = (u'hai mươi', u'ba mươi', u'bốn mươi', u'năm mươi',
        u'sáu mươi', u'bảy mươi', u'tám mươi', u'chín mươi')
denom = ('',
         u'nghìn', u'triệu', u'tỷ', u'nghìn tỷ', u'trăm nghìn tỷ',
         'Quintillion', 'Sextillion', 'Septillion', 'Octillion', 'Nonillion',
         'Decillion', 'Undecillion', 'Duodecillion', 'Tredecillion',
         'Quattuordecillion', 'Sexdecillion', 'Septendecillion',
         'Octodecillion', 'Novemdecillion', 'Vigintillion')


class Num2Word_VI(object):

    def _convert_nn(self, val):
        if val < 20:
            return to_19[val]
        for (dcap, dval) in ((k, 20 + (10 * v)) for (v, k) in enumerate(tens)):
            if dval + 10 > val:
                if val % 10:
                    a = u'lăm'
                    if to_19[val % 10] == u'một':
                        a = u'mốt'
                    else:
                        a = to_19[val % 10]
                    if to_19[val % 10] == u'năm':
                        a = u'lăm'
                    return dcap + ' ' + a
                return dcap

    def _convert_nnn(self, val):
        word = ''
        (mod, rem) = (val % 100, val // 100)
        if rem > 0:
            word = to_19[rem] + u' trăm'
            if mod > 0:
                word = word + ' '
        if mod > 0 and mod < 10:
            if mod == 5:
                word = word != '' and word + u'lẻ năm' or word + u'năm'
            else:
                word = word != '' and word + u'lẻ ' \
                    + self._convert_nn(mod) or word + self._convert_nn(mod)
        if mod >= 10:
            word = word + self._convert_nn(mod)
        return word

    def vietnam_number(self, val):
        if val < 100:
            return self._convert_nn(val)
        if val < 1000:
            return self._convert_nnn(val)
        for (didx, dval) in ((v - 1, 1000 ** v) for v in range(len(denom))):
            if dval > val:
                mod = 1000 ** didx
                lval = val // mod
                r = val - (lval * mod)

                ret = self._convert_nnn(lval) + u' ' + denom[didx]
                if 99 >= r > 0:
                    ret = self._convert_nnn(lval) + u' ' + denom[didx] + u' lẻ'
                if r > 0:
                    ret = ret + ' ' + self.vietnam_number(r)
                return ret

    def _parse_number(self, number):
        """Parse number from int, float, or string with thousand separators."""
        if isinstance(number, (int, float)):
            return number
        s = str(number)
        # 2+ groups of .xxx or ,xxx = thousand separators
        if re.search(r"(?:[.,]\d{3}).*[.,]\d{3}", s):
            return int(s.replace(".", "").replace(",", ""))
        # Has separator = decimal
        if "." in s or "," in s:
            return float(s.replace(",", "."))
        return int(s)

    def str_to_number(self, s):
        return self._parse_number(s)

    def number_to_text(self, number):
        parsed = self._parse_number(number)
        s = str(parsed)
        if '.' in s:
            int_part, dec_part = s.split('.', 1)
        else:
            int_part, dec_part = s, ''

        start_word = self.vietnam_number(int(int_part))
        final_result = start_word
        if dec_part and any(d != '0' for d in dec_part):
            final_result = final_result + ' phẩy '
            if len(dec_part) <= 2:
                final_result = final_result + self.vietnam_number(int(dec_part))
            else:
                final_result = final_result + ' '.join(to_19[int(d)] for d in dec_part)
        return final_result

    def to_cardinal(self, number):
        return self.number_to_text(number)

    def to_ordinal(self, number):
        return self.to_cardinal(number)
