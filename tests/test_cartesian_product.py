from simple_rl import cartesian_product


def test_cartesian_product():
    test_dict = {'a': 1, 'b': ['a', 'b']}
    cart_product = cartesian_product(test_dict)

    assert cart_product == [{'a': 1, 'b': 'a'}, {'a': 1, 'b': 'b'}]
