# This file is part of the NUS M2 scorer.
# The NUS M2 scorer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NUS M2 scorer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# file: util.py
#

import operator
import random
import math
import re

def smart_open(fname, mode = 'r'):
    if fname.endswith('.gz'):
        import gzip
        # Using max compression (9) by default seems to be slow.                                
        # Let's try using the fastest.                                                          
        return gzip.open(fname, mode, 1)
    else:
        return open(fname, mode, encoding='utf-8')


def randint(b, a=0):
    return random.randint(a,b)

def uniq(seq, idfun=None):
    # order preserving                                                                          
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:                                                               
        # if seen.has_key(marker)                                                               
        # but in new ones:                                                                      
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


def sort_dict(myDict, byValue=False, reverse=False):
    if byValue:
        items = list(myDict.items())
        items.sort(key = operator.itemgetter(1), reverse=reverse)
    else:
        items = sorted(myDict.items())
    return items

def max_dict(myDict, byValue=False):
    if byValue:
        skey=lambda x:x[1]
    else:
        skey=lambda x:x[0]
    return max(list(myDict.items()), key=skey)


def min_dict(myDict, byValue=False):
    if byValue:
        skey=lambda x:x[1]
    else:
        skey=lambda x:x[0]
    return min(list(myDict.items()), key=skey)

def paragraphs(lines, is_separator=lambda x : x == '\n', joiner=''.join):
    paragraph = []
    for line in lines:
        if is_separator(line):
            if paragraph:
                yield joiner(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield joiner(paragraph)


def isASCII(word):
    try:
        word = word.decode("ascii")
        return True
    except UnicodeEncodeError :
        return False
    except UnicodeDecodeError:
        return False


def intersect(x, y):
    return [z for z in x if z in y]



# Mapping Windows CP1252 Gremlins to Unicode
# from http://effbot.org/zone/unicode-gremlins.htm
cp1252 = {
    # from http://www.microsoft.com/typography/unicode/1252.htm
    "\x80": "\u20AC", # EURO SIGN
    "\x82": "\u201A", # SINGLE LOW-9 QUOTATION MARK
    "\x83": "\u0192", # LATIN SMALL LETTER F WITH HOOK
    "\x84": "\u201E", # DOUBLE LOW-9 QUOTATION MARK
    "\x85": "\u2026", # HORIZONTAL ELLIPSIS
    "\x86": "\u2020", # DAGGER
    "\x87": "\u2021", # DOUBLE DAGGER
    "\x88": "\u02C6", # MODIFIER LETTER CIRCUMFLEX ACCENT
    "\x89": "\u2030", # PER MILLE SIGN
    "\x8A": "\u0160", # LATIN CAPITAL LETTER S WITH CARON
    "\x8B": "\u2039", # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "\x8C": "\u0152", # LATIN CAPITAL LIGATURE OE
    "\x8E": "\u017D", # LATIN CAPITAL LETTER Z WITH CARON
    "\x91": "\u2018", # LEFT SINGLE QUOTATION MARK
    "\x92": "\u2019", # RIGHT SINGLE QUOTATION MARK
    "\x93": "\u201C", # LEFT DOUBLE QUOTATION MARK
    "\x94": "\u201D", # RIGHT DOUBLE QUOTATION MARK
    "\x95": "\u2022", # BULLET
    "\x96": "\u2013", # EN DASH
    "\x97": "\u2014", # EM DASH
    "\x98": "\u02DC", # SMALL TILDE
    "\x99": "\u2122", # TRADE MARK SIGN
    "\x9A": "\u0161", # LATIN SMALL LETTER S WITH CARON
    "\x9B": "\u203A", # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "\x9C": "\u0153", # LATIN SMALL LIGATURE OE
    "\x9E": "\u017E", # LATIN SMALL LETTER Z WITH CARON
    "\x9F": "\u0178", # LATIN CAPITAL LETTER Y WITH DIAERESIS
}

def fix_cp1252codes(text):
    # map cp1252 gremlins to real unicode characters
    if re.search("[\x80-\x9f]", text):
        def fixup(m):
            s = m.group(0)
            return cp1252.get(s, s)
        if isinstance(text, type("")):
            # make sure we have a unicode string
            text = str(text, "iso-8859-1")
        text = re.sub("[\x80-\x9f]", fixup, text)
    return text

def clean_utf8(text):
    return [x for x in text if x > '\x1f' and x < '\x7f']

def pairs(iterable, overlapping=False):
    iterator = iterable.__iter__()
    token = next(iterator)
    i = 0
    for lookahead in iterator:
        if overlapping or i % 2 == 0: 
            yield (token, lookahead)
        token = lookahead
        i += 1
    if i % 2 == 0:
        yield (token, None)

def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
        
    return L

def softmax(values):
    a = max(values)
    Z = 0.0
    for v in values:
        Z += math.exp(v - a)
    sm = [math.exp(v-a) / Z for v in values]
    return sm
