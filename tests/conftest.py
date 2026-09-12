"""Run `async def` tests without pytest-asyncio.

Ten lines of hook beat a plugin dependency. Each coroutine test gets its own
event loop, which is what we want anyway: a leaked socket in one test must not
reach the next one.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None
    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(func(**kwargs))
    return True
