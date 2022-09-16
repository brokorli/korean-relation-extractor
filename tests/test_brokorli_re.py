import unittest
from brokorli_re import BrokorliRE


class BrokorliReTest(unittest.TestCase):

    sentence = "아이유는 1993년 5월 16일에 태어났다."
    subj = "아이유"
    obj = "1993년 5월 16일"

    def __init__(self, *args, **kwargs):
        super(BrokorliReTest, self).__init__(*args, **kwargs)

        self.brokorli_re = BrokorliRE()

    def test_runs(self):
        self.brokorli_re.extract(sentence=self.sentence, subj=self.subj, obj=self.obj)


if __name__ == "__main__":
    unittest.main()
