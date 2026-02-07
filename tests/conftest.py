import pytest
import pygmu2 as pg


@pytest.fixture(autouse=True)
def _set_sample_rate():
    pg.set_sample_rate(44100)
    yield
    pg.set_sample_rate(44100)
