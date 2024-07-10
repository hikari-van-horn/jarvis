"""Hello unit test module."""

from memory.hello import hello


def test_hello():
    """Test the hello function."""
    assert hello() == "Hello memory"
