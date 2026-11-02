"""Owner-only directory protection on Windows, decided from the security descriptor.

POSIX ``mkdir(mode=0o700)`` has no Windows equivalent that every supported Python applies
(CPython applies a protected ACL for ``0o700`` only from 3.12.4/3.13), so the database
directory is restricted explicitly (``src.state.db``) and then **verified** here. The
decision is made from the directory's security descriptor in SDDL, which Windows produces
with SIDs or fixed two-letter aliases: never from localized account names, ``icacls``
output or the number of entries (Session 10 review, P2).

Threat model (owner-only, the counterpart of POSIX ``0700``): the DACL must be protected
(no inherited entries) and may allow access only to

* the current user;
* ``SYSTEM`` (S-1-5-18) and ``BUILTIN\\Administrators`` (S-1-5-32-544), which can take
  ownership of, back up or restore any file regardless of its DACL, like root on POSIX;
* ``OWNER RIGHTS`` (S-1-3-4), only while the owner is one of the principals above.

Any other allowed trustee (Everyone, Users, Authenticated Users, Guests, another account,
an unknown alias or an object/callback ACE type) makes the directory not owner-only.
Deny entries only remove access and are accepted. The current user must have full control,
directly or through ``OWNER RIGHTS`` as owner. A missing or null DACL is rejected.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

SYSTEM = "S-1-5-18"
ADMINISTRATORS = "S-1-5-32-544"
OWNER_RIGHTS = "S-1-3-4"
FILE_ALL_ACCESS = 0x001F01FF
GENERIC_ALL = 0x10000000

# SDDL SID aliases (https://learn.microsoft.com/windows/win32/secauthz/sid-strings).
WELL_KNOWN = {
    "AN": "S-1-5-7",
    "AO": "S-1-5-32-548",
    "AU": "S-1-5-11",
    "BA": ADMINISTRATORS,
    "BG": "S-1-5-32-546",
    "BO": "S-1-5-32-551",
    "BU": "S-1-5-32-545",
    "CO": "S-1-3-0",
    "CG": "S-1-3-1",
    "IU": "S-1-5-4",
    "LS": "S-1-5-19",
    "NS": "S-1-5-20",
    "NU": "S-1-5-2",
    "OW": OWNER_RIGHTS,
    "PO": "S-1-5-32-550",
    "PS": "S-1-5-10",
    "PU": "S-1-5-32-547",
    "RC": "S-1-5-12",
    "RD": "S-1-5-32-555",
    "RE": "S-1-5-32-552",
    "RU": "S-1-5-32-554",
    "SO": "S-1-5-32-549",
    "SU": "S-1-5-6",
    "SY": SYSTEM,
    "WD": "S-1-1-0",
    "AC": "S-1-15-2-1",
    "WR": "S-1-5-33",
    "ER": "S-1-5-32-573",
    "HA": "S-1-5-32-578",
}
# Aliases relative to the account's domain: <domain SID>-<RID>.
DOMAIN_RELATIVE = {
    "LA": 500,
    "LG": 501,
    "DA": 512,
    "DU": 513,
    "DG": 514,
    "DC": 515,
    "DD": 516,
    "CA": 517,
    "SA": 518,
    "EA": 519,
    "PA": 520,
}
RIGHTS = {
    "GA": GENERIC_ALL,
    "GR": 0x80000000,
    "GW": 0x40000000,
    "GX": 0x20000000,
    "RC": 0x00020000,
    "SD": 0x00010000,
    "WD": 0x00040000,
    "WO": 0x00080000,
    "RP": 0x10,
    "WP": 0x20,
    "CC": 0x1,
    "DC": 0x2,
    "LC": 0x4,
    "SW": 0x8,
    "LO": 0x80,
    "DT": 0x40,
    "CR": 0x100,
    "FA": FILE_ALL_ACCESS,
    "FR": 0x00120089,
    "FW": 0x00120116,
    "FX": 0x001200A0,
}


@dataclass(frozen=True)
class Ace:
    kind: str  # "A" allow, "D" deny, anything else is not accepted
    flags: frozenset[str]
    rights: int | None  # None: not parseable
    sid: str  # resolved SID ("S-1-...") or "?<alias>" when unknown


@dataclass(frozen=True)
class SecurityDescriptor:
    owner: str | None
    dacl_present: bool
    dacl_protected: bool
    null_dacl: bool
    aces: tuple[Ace, ...]


def _resolve(token: str, user_sid: str) -> str:
    token = token.strip()
    if token.upper().startswith("S-1-"):
        return token.upper()
    if token in WELL_KNOWN:
        return WELL_KNOWN[token]
    if token in DOMAIN_RELATIVE and user_sid.startswith("S-1-5-21-"):
        return f"{user_sid.rsplit('-', 1)[0]}-{DOMAIN_RELATIVE[token]}"
    return f"?{token}"


def _rights(text: str) -> int | None:
    text = text.strip()
    if text.lower().startswith("0x"):
        try:
            return int(text, 16)
        except ValueError:
            return None
    if text.isdigit():
        return int(text)
    if len(text) % 2:
        return None
    mask = 0
    for i in range(0, len(text), 2):
        code = text[i : i + 2]
        if code not in RIGHTS:
            return None
        mask |= RIGHTS[code]
    return mask


SDDL_RE = re.compile(
    r"^(?:O:(?P<owner>[^:()]+?)(?=[GDS]:|$))?"
    r"(?:G:(?P<group>[^:()]+?)(?=[DS]:|$))?"
    r"(?:D:(?P<dflags>[A-Z_]*)(?P<aces>(?:\([^()]*\))*))?"
    r"(?:S:.*)?$"
)


def parse_sddl(sddl: str, user_sid: str) -> SecurityDescriptor:
    m = SDDL_RE.match(sddl.strip())
    if m is None:
        raise ValueError(f"unrecognised security descriptor: {sddl!r}")
    dflags = m.group("dflags")
    aces = []
    for body in re.findall(r"\(([^()]*)\)", m.group("aces") or ""):
        parts = body.split(";")
        if len(parts) < 6:
            raise ValueError(f"unrecognised ACE: ({body})")
        kind, flags, rights, _, _, sid = parts[:6]
        flag_set = frozenset(flags[i : i + 2] for i in range(0, len(flags), 2))
        aces.append(Ace(kind, flag_set, _rights(rights), _resolve(sid, user_sid)))
    owner = m.group("owner")
    return SecurityDescriptor(
        owner=_resolve(owner, user_sid) if owner else None,
        dacl_present=dflags is not None,
        dacl_protected=dflags is not None and "P" in dflags.replace("NO_ACCESS_CONTROL", ""),
        null_dacl=dflags is not None and "NO_ACCESS_CONTROL" in dflags,
        aces=tuple(aces),
    )


def _full(rights: int | None) -> bool:
    if rights is None:
        return False
    return bool(rights & GENERIC_ALL) or rights & FILE_ALL_ACCESS == FILE_ALL_ACCESS


def owner_only_problems(sddl: str, user_sid: str) -> list[str]:
    """Why the descriptor is not owner-only (empty list: owner-only), per the module's
    threat model. ``user_sid`` is the current account's SID (``S-1-5-21-…``)."""
    user = user_sid.upper()
    try:
        sd = parse_sddl(sddl, user)
    except ValueError as exc:
        return [str(exc)]
    privileged = {user, SYSTEM, ADMINISTRATORS}
    problems: list[str] = []
    if not sd.dacl_present or sd.null_dacl:
        return ["no DACL (everyone has access)"]
    if not sd.dacl_protected:
        problems.append("the DACL is not protected: entries are inherited from the parent")
    if sd.owner not in privileged:
        problems.append(f"the owner {sd.owner} is not the current user, SYSTEM or Administrators")
    user_full = False
    for ace in sd.aces:
        if "ID" in ace.flags:
            problems.append(f"inherited entry for {ace.sid}")
        if ace.kind == "D":
            continue
        if ace.kind != "A":
            problems.append(f"entry type {ace.kind!r} for {ace.sid} is not accepted")
            continue
        if ace.sid == OWNER_RIGHTS:
            if sd.owner == user and "IO" not in ace.flags and _full(ace.rights):
                user_full = True
            continue  # the owner is checked above
        if ace.sid not in privileged:
            problems.append(
                f"access is allowed to {ace.sid}, which is not the current user, "
                "SYSTEM or Administrators"
            )
            continue
        if ace.sid == user and "IO" not in ace.flags and _full(ace.rights):
            user_full = True
    if not user_full:
        problems.append("the current user has no full control of the directory")
    return problems


def windows_sddl(path: Path) -> str:
    """The owner and DACL of ``path`` in SDDL (Windows only; ``GetNamedSecurityInfoW``)."""
    if sys.platform != "win32":
        raise OSError("security descriptors are read on Windows only")
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    info = 0x1 | 0x4  # OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION
    get = advapi32.GetNamedSecurityInfoW
    get.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        *(ctypes.POINTER(ctypes.c_void_p),) * 5,
    ]
    get.restype = wintypes.DWORD
    to_text = advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW
    to_text.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(wintypes.ULONG),
    ]
    to_text.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    owner, dacl, sd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    rc = get(
        str(path), 1, info, ctypes.byref(owner), None, ctypes.byref(dacl), None, ctypes.byref(sd)
    )  # SE_FILE_OBJECT = 1
    if rc != 0:
        raise ctypes.WinError(rc)
    try:
        text = wintypes.LPWSTR()
        if not to_text(sd, 1, info, ctypes.byref(text), None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return str(text.value)
        finally:
            kernel32.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        kernel32.LocalFree(sd)
