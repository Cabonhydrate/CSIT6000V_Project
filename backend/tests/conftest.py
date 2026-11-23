import os
import tempfile
from pathlib import Path

# Set isolated data dir before importing any backend module.
_TEST_TEMP=tempfile.TemporaryDirectory(prefix='paper-tutor-tests-')
os.environ['APP_DATA_DIR']=_TEST_TEMP.name

import pytest
from app import db


@pytest.fixture(autouse=True)
def fresh_db():
    db.initialize()
    with db.connect() as c:
        for table in ('run_events','messages','runs','eval_cases','chunks','conversations','papers','memories'):
            c.execute(f'DELETE FROM {table}')
    yield
