"""Synthetic unit tests for reference-entry anchored Tradingade exit diagnostics.

No private market/export datasets required. The tests intentionally do NOT claim
full external Strategy Builder or causal execution parity.
"""
import unittest
from tradingade_exit_parity import (
    ExitMachine, RenkoBrick, StopSettings, complete_source_candles,
    reconstruct_renko_with_wicks, checked_reference_flips,
    predict_exit,parse_ms,analyze_manifest,
)


class TestStopEngine(unittest.TestCase):
    def setUp(self):
        self.config=StopSettings(.009,.008)

    def test_unarmed_fixed_stop_short_even_if_trail_would_be_tighter(self):
        state=ExitMachine(0,100.0,'short',self.config)
        hit=state.advance(RenkoBrick(300_000,100,100.91,100,100.9))
        self.assertEqual(hit.trigger,'fixed_stop')
        self.assertAlmostEqual(hit.stop_level,100.9,places=6)
        self.assertEqual(hit.ts_ms,300_000)

    def test_trailing_stop_long_activates_after_confirmed_favorable_brick(self):
        state=ExitMachine(0,100.0,'long',self.config)
        self.assertIsNone(state.advance(RenkoBrick(300_000,100,101,100,101)))
        self.assertTrue(state.armed)
        self.assertAlmostEqual(state.stop_level()[0],100.192)
        hit=state.advance(RenkoBrick(600_000,101,101.1,100.1,100.3))
        self.assertEqual(hit.trigger,'armed_trailing')
        self.assertAlmostEqual(hit.stop_level,100.192)
        with self.assertRaises(RuntimeError):state.advance(RenkoBrick(900_000,100,100.1,99,99))

    def test_same_brick_high_low_do_not_retroactively_activate_stop(self):
        state=ExitMachine(0,100.0,'long',self.config)
        # 99.5 < newly computed 100.192 trail, but cannot use NEW high before
        # the intrabar low without evidence of intrabar path. Fixed 99.1 holds.
        self.assertIsNone(state.advance(RenkoBrick(300_000,100,101,99.5,100.5)))
        self.assertTrue(state.armed)
        self.assertEqual(state.advance(RenkoBrick(600_000,100.5,100.6,100,100.1)).trigger,'armed_trailing')

    def test_reference_entry_brick_is_excluded_from_lookback(self):
        bricks=[RenkoBrick(300_000,100,100.1,99.0,99.0),RenkoBrick(600_000,99,99.2,98.9,99)]
        pred=predict_exit(bricks,[b.ts_ms for b in bricks],300_000,100,'long',self.config)
        self.assertEqual(pred.ts_ms,600_000)
        self.assertEqual(pred.trigger,'fixed_stop')

    def test_two_box_reversal_and_unique_brick_milliseconds(self):
        bricks=reconstruct_renko_with_wicks([(0,100.0),(300,100.3),(600,100.2),(900,100.1)])
        self.assertEqual([(b.ts_ms,b.open,b.close) for b in bricks[:3]],
                         [(300000,100.0,100.1),(300001,100.1,100.2),(300002,100.2,100.3)])
        self.assertEqual(bricks[-1].close,100.1)
        self.assertEqual(bricks[-1].open,100.2)
        self.assertEqual(bricks[-1].ts_ms,900000)

    def test_complete_source_rejects_gaps_and_partial_buckets(self):
        sample=[(ts,100 + ts/10000) for ts in [0,60,120,180,240,300,360,480,540]]
        self.assertEqual(complete_source_candles(sample,300),{0:100+.024})
        self.assertEqual(complete_source_candles(sample,3600),{})

    def test_invalid_event_and_parameter_bounds_fail_closed(self):
        with self.assertRaises(ValueError):parse_ms('2026-10-02T13:15:00')
        with self.assertRaises(ValueError):StopSettings(0,.008)
        with self.assertRaises(ValueError):StopSettings(.009,float('nan'))
        st=ExitMachine(0,100,'long',self.config)
        st.advance(RenkoBrick(300000,100,101,100,101))
        with self.assertRaises(ValueError):st.advance(RenkoBrick(300000,100,101,100,101))

    def test_flip_continuity_uses_reference_ledger_only(self):
        rec={'entryUtc':'2026-10-17T00:00:00.000Z','side':'short','entryPrice':100,
             'exitUtc':'2026-10-17T00:05:00.000Z','exitSignal':'long','exitPrice':101}
        opposite={'entryUtc':rec['exitUtc'],'side':'long','entryPrice':101,'exitUtc':'2026-10-17T01:00:00.000Z','exitSignal':'SL','exitPrice':99}
        found=checked_reference_flips({'trades':{'closed':[rec,opposite],'open':[]}})
        self.assertEqual(found,{'reference_flip_exits':1,'continuing_opposite_entries':1,'unpaired_flip_exits':0})

    def test_reference_exit_is_never_an_input_to_prediction(self):
        import copy
        settings=[['💰 Take Profit Type','Trailing TP %'],
                  ['📛 Stop Loss Type','SL %'],['🚀TTP  %','0']]
        trade={'number':1,'entryUtc':'1970-01-01T00:00:00Z',
               'entryPrice':100,'side':'long','exitUtc':'1970-01-01T00:10:00Z',
               'exitPrice':100.192,'exitSignal':'TP'}
        seed={'id':'tvseed-test','instrument':{'symbol':'BINANCE:SOLUSDT'},
              'candidateGenes':{'stopLossBps':90,'trailingDeviationBps':80},
              'originalSettings':{'safePropertyRows':settings},
              'trades':{'closed':[trade],'open':[]}}
        payload={'schemaVersion':2,'sourceTimeZone':'Europe/Berlin',
                 'secretsExcluded':True,'seeds':[seed]}
        bricks=[RenkoBrick(0,100,100,100,100),
                RenkoBrick(300000,100,101,100,101),
                RenkoBrick(600000,101,101,100.1,100.2)]
        first=analyze_manifest(payload,bricks,1_200_000)[1][0]
        altered=copy.deepcopy(payload)
        altered['seeds'][0]['trades']['closed'][0]['exitUtc']='1970-01-01T00:15:00Z'
        altered['seeds'][0]['trades']['closed'][0]['exitPrice']=99.99
        second=analyze_manifest(altered,bricks,1_200_000)[1][0]
        self.assertEqual(first['predictedExitUtc'],second['predictedExitUtc'])
        self.assertEqual(first['predictedExitPx'],second['predictedExitPx'])
        self.assertNotEqual(first['timeDeltaMs'],second['timeDeltaMs'])

    def test_legacy_or_incomplete_manifest_fails(self):
        with self.assertRaises(ValueError):analyze_manifest({'schemaVersion':1},[],0)


if __name__=='__main__':unittest.main()