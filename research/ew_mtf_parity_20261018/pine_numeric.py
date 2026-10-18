"""Pine float arithmetic with comparison operands rounded to nine decimals.

Reference: https://www.tradingview.com/pine-script-docs/language/type-system/
No quantization during arithmetic; only comparisons round operands. Values in
this replay are finite prices/ratios; missing input is handled before wrapping.
"""
import math


class PineFloat(float):
    def __add__(self, other):
        return PineFloat(float(self)+float(other))

    __radd__ = __add__

    def __sub__(self, other):
        return PineFloat(float(self)-float(other))

    def __rsub__(self, other):
        return PineFloat(float(other)-float(self))

    def __mul__(self, other):
        return PineFloat(float(self)*float(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return PineFloat(float(self)/float(other))

    def __rtruediv__(self, other):
        return PineFloat(float(other)/float(self))

    def __abs__(self):
        return PineFloat(abs(float(self)))

    def __neg__(self):
        return PineFloat(-float(self))

    def __lt__(self, other):
        return round(float(self),9)<round(float(other),9)

    def __le__(self, other):
        return round(float(self),9)<=round(float(other),9)

    def __gt__(self, other):
        return round(float(self),9)>round(float(other),9)

    def __ge__(self, other):
        return round(float(self),9)>=round(float(other),9)

    def __eq__(self, other):
        return round(float(self),9)==round(float(other),9)

    def __ne__(self, other):
        return math.isfinite(self) and math.isfinite(other) and not self==other

