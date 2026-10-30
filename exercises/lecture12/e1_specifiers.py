"""Run from the project root: python exercises/lecture12/e1_specifiers.py"""
from packaging.specifiers import SpecifierSet

releases = ["0.21.1", "1.0.0", "1.0.1", "1.1.0", "1.2.4"]
later = ["1.10.0", "2.0.0"]
for spec in ["~=1.0.1", "~=1.0", "==1.*", "!=1.0.*", ">0.21,<=1.1"]:
    s = SpecifierSet(spec)
    print(f"{spec:<12} {list(s.filter(releases))}  later: {list(s.filter(later))}")
