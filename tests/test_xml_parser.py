import unittest

from xml_parser import parse_decipher_xml


class ConditionPropagationTest(unittest.TestCase):
    def test_combines_block_loop_question_and_nested_conditions(self):
        xml = """
        <survey alt="Condition fixture">
            <block cond="block_cond">
                <radio label="Q_block">
                    <title>Block condition only</title>
                    <row label="r1">Yes</row>
                </radio>
                <radio label="Q_block_question" cond="question_cond">
                    <title>Block and question condition</title>
                    <row label="r1">Yes</row>
                </radio>
                <loop label="loop1" cond="loop_cond">
                    <radio label="Q_block_loop">
                        <title>Block and loop condition</title>
                        <row label="r1">Yes</row>
                    </radio>
                    <radio label="Q_all" cond="question_cond">
                        <title>Block, loop, and question condition</title>
                        <row label="r1">Yes</row>
                    </radio>
                </loop>
                <block cond="nested_block_cond">
                    <radio label="Q_nested_block">
                        <title>Nested block condition</title>
                        <row label="r1">Yes</row>
                    </radio>
                </block>
            </block>
            <loop label="loop2" cond="outer_loop_cond">
                <radio label="Q_loop_question" cond="question_cond">
                    <title>Loop and question condition</title>
                    <row label="r1">Yes</row>
                </radio>
            </loop>
        </survey>
        """

        parsed = parse_decipher_xml(xml)
        cond_by_label = {q["label"]: q["cond"] for q in parsed["questions"]}

        self.assertEqual(cond_by_label["Q_block"], "block_cond")
        self.assertEqual(
            cond_by_label["Q_block_question"],
            "(block_cond) and (question_cond)",
        )
        self.assertEqual(
            cond_by_label["Q_block_loop"],
            "(block_cond) and (loop_cond)",
        )
        self.assertEqual(
            cond_by_label["Q_all"],
            "((block_cond) and (loop_cond)) and (question_cond)",
        )
        self.assertEqual(
            cond_by_label["Q_nested_block"],
            "(block_cond) and (nested_block_cond)",
        )
        self.assertEqual(
            cond_by_label["Q_loop_question"],
            "(outer_loop_cond) and (question_cond)",
        )


if __name__ == "__main__":
    unittest.main()
