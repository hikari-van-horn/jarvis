"""Hello unit test module."""

from vision.hello import hello


def test_hello():
    """Test the hello function."""
    assert hello() == "Hello vision"
