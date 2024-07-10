"""Hello unit test module."""

from jarvis.hello import hello


def test_hello():
    """Test the hello function."""
    assert hello() == "Hello jarvis"
