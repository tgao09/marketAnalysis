import unittest
from pathlib import Path
from rl_portfolio_management.generate_report import main

class ReportSmokeTest(unittest.TestCase):
    def test_self_contained_report(self):
        main(); root=Path(__file__).parents[1]; html=(root/'report/report.html').read_text(encoding='utf-8')
        for section in ['Executive conclusion','Walk-forward evidence','Leakage and execution safeguards','Optimization and sensitivity','Per-ticker attribution','Regime analysis','Reproduction']:
            self.assertIn(section,html)
        self.assertIn('data:image/png;base64,',html)
        self.assertNotRegex(html,r'''(?:src|href)=["']https?://''')
        self.assertTrue((root/'report/report_data.json').is_file())

if __name__=='__main__': unittest.main()
