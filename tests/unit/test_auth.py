from auth import API_KEY_PREFIX, generate_api_key, hash_api_key


def test_generated_key_has_prefix_and_is_random() -> None:
    a = generate_api_key()
    b = generate_api_key()
    assert a.startswith(API_KEY_PREFIX)
    assert b.startswith(API_KEY_PREFIX)
    assert a != b
    assert len(a) > 40


def test_hash_is_stable_and_64_hex_chars() -> None:
    h = hash_api_key("hello")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
    assert hash_api_key("hello") == h
    assert hash_api_key("world") != h
