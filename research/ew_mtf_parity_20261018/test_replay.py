import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pine_numeric import PineFloat
from ew_replay import Pivot, Config, add_pivot, best_pattern, analyze_htf, pivot_series, replay_export

ROOT=Path(__file__).resolve().parent

class PineSemanticsTests(unittest.TestCase):
    def test_comparison_rounds_but_arithmetic_retains_precision(self):
        value=PineFloat(.3)-PineFloat(.2)
        self.assertEqual(value,.1)
        self.assertFalse(value<.1)
        self.assertNotEqual(float(value),.1)
        self.assertFalse(PineFloat(float('nan')) != 0)

    def test_exact_eighty_percent_is_flat_not_zigzag(self):
        arr=[]
        for i,(price,hi) in enumerate([(117.9,True),(116.4,False),(117.6,True),(116.80000000000001,False)]):
            add_pivot(arr,price,hi,i,.1,1)
        pat=best_pattern(arr)
        self.assertEqual(pat.kind,'regular flat')
        self.assertEqual(pat.confidence,.4)

    def test_pivot_confirmation_and_rightmost_tie(self):
        x=np.array([0,1,3,3,2,1,0.])
        out=pivot_series(x,2,True)
        self.assertEqual(np.flatnonzero(np.isfinite(out)).tolist(),[5])
        self.assertEqual(out[5],3)

    def test_extension_does_not_push(self):
        arr=[]
        self.assertEqual(add_pivot(arr,100,True,1,1,2.5),2)
        self.assertEqual(add_pivot(arr,100+1e-12,True,2,1,2.5),0)
        self.assertEqual(add_pivot(arr,101,True,3,1,2.5),1)
        self.assertEqual(add_pivot(arr,99,False,4,1,2.5),0)
        self.assertEqual(add_pivot(arr,98.5,False,5,1,2.5),2)

    def test_correction_implies_opposite_htf_trend(self):
        arr=[]
        for i,(price,hi) in enumerate([(100,True),(90,False),(95,True),(85,False)]):
            add_pivot(arr,price,hi,i,1,1)
        ctx=analyze_htf(arr)
        self.assertEqual(ctx.kind,'zigzag')
        self.assertEqual(ctx.bias,1)
        self.assertTrue(ctx.bull)

class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=pd.read_csv(ROOT/'inputs/BINANCE_SOLUSDT, 5 (5).csv',float_precision='round_trip')
        cls.replay=replay_export(cls.frame)

    def test_reference_outputs_cannot_leak_into_replay(self):
        bad=self.frame.copy()
        for col in self.replay:
            bad[col]=99999
        pd.testing.assert_frame_equal(replay_export(bad),self.replay)

    def test_no_future_dependency(self):
        for end in [250,850,5000]:
            pd.testing.assert_frame_equal(replay_export(self.frame.iloc[:end]),self.replay.iloc[:end])

    def test_fixed_strategy_window_signal_parity(self):
        mask=self.frame.time>=pd.Timestamp('2026-09-07T00:00:00Z').timestamp()
        for col in ['Long Entry','Short Entry','Wave 5 Exit (bullish impulse)','Wave 5 Exit (bearish impulse)']:
            np.testing.assert_array_equal(self.replay.loc[mask,col],self.frame.loc[mask,col])

    def test_all_mtf2_states_match_on_entire_export(self):
        for col in self.replay:
            if col.startswith('Audit htf2'):
                np.testing.assert_allclose(self.replay[col],self.frame[col],rtol=0,atol=1e-10)

if __name__=='__main__':
    unittest.main()
