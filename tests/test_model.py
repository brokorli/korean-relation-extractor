import unittest
from brokorli_re.model import ReModel


# TestCase를 작성
class ModelTest(unittest.TestCase):

    sentence = "아이유는 1993년 5월 16일에 태어났다."
    subj = "아이유"
    obj = "1993년 5월 16일"

    def __init__(self, *args, **kwargs):
        super(ModelTest, self).__init__(*args, **kwargs)

        self.model = ReModel()

    def test_prompt(self):
        prompt = ReModel.make_prompt(
            sentence=self.sentence, subj="아이유", obj="1993년 5월 16일"
        )

        self.assertEqual(
            prompt,
            "아이유는 1993년 5월 16일에 태어났다. [SEP] [E1] 아이유 [/E1] [MASK] [E2] 1993년 5월 16일 [/E2]",
        )

    def test_tokenize(self):
        ReModel.make_prompt(sentence=self.sentence, subj=self.subj, obj=self.obj)

    def test_inference(self):
        self.model(sentence=self.sentence, subj=self.subj, obj=self.obj)


if __name__ == "__main__":
    unittest.main()
