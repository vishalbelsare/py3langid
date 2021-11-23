import langid


def test_langid():
    '''Test if the language detection functions work'''
    assert langid.classify('This text is in English.')[0] == 'en'
    assert langid.rank('This text is in English.')[0][0] == 'en'
