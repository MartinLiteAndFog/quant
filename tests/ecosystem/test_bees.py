"""ECO-08 local/hive/hybrid bees: genes, lineage, safe budgets, no live orders."""
from dataclasses import replace
from datetime import date
from decimal import Decimal as D
import random
import unittest

from quant.ecosystem.breeding.bees import (
    Bee, BreedingError, GeneSpec, Genome, Plant,
    bee_honey, breed_genomes, fund_local_seeds, mutate_bee,
    pollinate, spec_catalog, validate_genome,
)

DAY=date(2026,11,1)


def catalog():
    return [
        GeneSpec("protective_stop","immutable_safety_policy","switch",default=True),
        GeneSpec("max_leverage","required_bounded_financial","number",1,10,.1,3),
        GeneSpec("max_margin_fraction","required_bounded_financial","number",.01,.9,.01,.5),
        GeneSpec("reproduction_budget","required_bounded_financial","number",.08,.15,.01,.10),
        GeneSpec("sensitivity","required_axis","number",12,25,.1,17.2),
        GeneSpec("ttp","required_axis","number",.25,1.65,.05,.65),
        GeneSpec("rsi_length","optional_strategy_gene","number",5,50,1,14),
        GeneSpec("adx_gate","optional_strategy_gene","switch",default=False),
    ]


def gene(name, sens=17.2, ttp=.65, optional=None, tile="sol-imba", module="imba"):
    values=dict(
        protective_stop=True,max_leverage=3.0,max_margin_fraction=.5,
        reproduction_budget=.10,sensitivity=sens,ttp=ttp,
    )
    if optional is not None:
        values.update(optional)
    return Genome(name+"-genome",1,module,tile,tuple(sorted(values.items())))


def plant(name,sens=17.2,ttp=.65,optional=None,tile="sol-imba",module="imba"):
    g=gene(name,sens,ttp,optional,tile,module)
    x=(sens-12)/13
    y=(ttp-.25)/1.4
    return Plant(name,g,x,y,tile,.7)


def bee(name,breed="mixer",tile="sol-imba",**opts):
    return Bee(name,breed,tile,**opts)


class BeeBreedingTests(unittest.TestCase):
    def setUp(self):
        self.specs=catalog()
        self.p=[plant("a",17.2,.65,{"rsi_length":14}),
                plant("b",19.2,.85,{"rsi_length":18})]

    def test_mixer_averages_actual_axis_genes_and_keeps_safety(self):
        child=breed_genomes(
            self.p[0].genome,self.p[1].genome,bee("mixer",mutation_strength=0),
            self.specs,random.Random(1),child_id="new-genome",target_tile="sol-imba",
        )
        self.assertAlmostEqual(child.values()["sensitivity"],18.2)
        self.assertAlmostEqual(child.values()["ttp"],.75)
        self.assertTrue(child.values()["protective_stop"])
        self.assertIn("reproduction_budget", child.values())
        self.assertIn("max_leverage", child.values())
        self.assertEqual(child.parent_genomes,
                         ("a-genome","b-genome"))

    def test_seed_coordinates_are_true_axis_genes(self):
        proposal=pollinate(DAY,10,[bee("b",search_radius=1,
                                        mutation_strength=.25)],
                          self.p,self.specs,mode="local")[0]
        values=proposal.child_genome.values()
        self.assertAlmostEqual(proposal.normalized_x,(values["sensitivity"]-12)/13)
        self.assertAlmostEqual(proposal.normalized_y,(values["ttp"]-.25)/1.4)
        self.assertEqual(proposal.destination_tile_id,"sol-imba")
        self.assertFalse(proposal.research_only)
        self.assertFalse(proposal.funded)
        self.assertEqual(proposal.lineage[2],"b")

    def test_local_hive_hybrid_distinct_and_reproducible(self):
        bees=[bee("mixer","mixer",hive_share_probability=1),
              bee("explorer","explorer",hive_share_probability=1)]
        sister=plant("distant",20,.95,tile="btc-other",module="imba")
        local=pollinate(DAY,42,bees,self.p+[sister],self.specs,mode="local")
        hive=pollinate(DAY,42,bees,self.p+[sister],self.specs,mode="central")
        hybrid=pollinate(DAY,42,bees,self.p+[sister],self.specs,mode="hybrid")
        self.assertEqual(len(local),2)
        self.assertEqual(len(hive),2)
        self.assertEqual(len(hybrid),4)
        self.assertEqual(hybrid,pollinate(DAY,42,list(reversed(bees)),
                                          list(reversed(self.p+[sister])),
                                          self.specs,mode="hybrid"))
        self.assertTrue(all(x.research_only for x in hive))
        self.assertTrue(all(not x.research_only for x in local))
        self.assertEqual(len({x.seed_id for x in hybrid}),4)
        self.assertTrue(all(x.child_genome.module_family=="imba" for x in hybrid))

    def test_add_drop_only_optional_and_protected_never_lost(self):
        no_optional=[plant("plain"),plant("other",19.2,.85)]
        b=bee("prune","pruner",optional_gene_add_rate=1,
              optional_gene_drop_rate=.1)
        seeds=pollinate(DAY,888,[b],no_optional,self.specs)
        for item in seeds:
            v=item.child_genome.values()
            self.assertTrue(v["protective_stop"])
            self.assertTrue({"max_leverage","max_margin_fraction",
                             "reproduction_budget","sensitivity","ttp"} <= set(v))
            validate_genome(item.child_genome,self.specs)
        with self.assertRaisesRegex(BreedingError,"missing protected"):
            validate_genome(Genome("broken",1,"imba","sol-imba",
                        (("sensitivity",17.2),("ttp",.65))),self.specs)
        with self.assertRaisesRegex(BreedingError,"immutable"):
            invalid=dict(self.p[0].genome.values());invalid["protective_stop"]=False
            validate_genome(Genome("evil",1,"imba","sol-imba",
                                   tuple(invalid.items())),self.specs)
        extra=bee("collector","collector",optional_gene_add_rate=1,
                  optional_gene_drop_rate=0)
        result=pollinate(DAY,5,[extra],no_optional,self.specs)[0]
        self.assertIn("rsi_length",result.child_genome.values())

    def test_capital_conservation_only_local_seed_and_parent_quota(self):
        children=pollinate(DAY,100,[bee("worker",hive_share_probability=1)],self.p,self.specs,mode="hybrid")
        funded, claims=fund_local_seeds(children,{"a":D("30"),"b":D("30")},
                        {"a":D("2"),"b":D("0")},
                        {"a":D(".10"),"b":D(".10")},child_share=".02")
        self.assertEqual(sum(claims.values()),D("60"))
        local=[x for x in funded if x.source=="local"]
        hive=[x for x in funded if x.source=="hive"]
        self.assertEqual(len(local),1)
        self.assertEqual(len(hive),1)
        self.assertTrue(local[0].funded)
        self.assertEqual(local[0].allocated_budget,D(".60"))
        self.assertFalse(hive[0].funded)
        self.assertEqual(hive[0].allocated_budget,0)
        with self.assertRaisesRegex(BreedingError,"duplicate seed"):
            fund_local_seeds(children+children,{"a":D("30"),"b":D("30")},{},
                             {"a":D(".10"),"b":D(".10")})
        with self.assertRaisesRegex(BreedingError,"locked margin"):
            fund_local_seeds(children,{"a":D("30"),"b":D("30")},
                             {"a":D("30"),"b":D("30")},
                             {"a":D(".10"),"b":D(".10")})

    def test_five_bee_breeds_and_evolved_search_parameters(self):
        breeds=["mixer","mosaic","explorer","collector","pruner"]
        for k,name in enumerate(breeds):
            agent=bee("b"+str(k),name)
            mutated=mutate_bee(agent,42,DAY)
            self.assertEqual(mutated,mutate_bee(agent,42,DAY))
            self.assertEqual(mutated.breed,agent.breed)
            self.assertLessEqual(mutated.optional_gene_drop_rate,.1)
            seeds=pollinate(DAY,42,[mutated],self.p,self.specs)
            self.assertEqual(len(seeds),1)

    def test_bee_honey_requires_held_out_results(self):
        untested=bee_honey("b",[("child1",3.0,False)])
        self.assertEqual(untested.verified_offspring,0)
        self.assertIsNone(untested.mean_out_of_sample_score)
        good=bee_honey("b",[("c1",.20,True),("c2",.40,True),
                                 ("c3",900,False)])
        self.assertEqual(good.verified_offspring,2)
        self.assertAlmostEqual(good.mean_out_of_sample_score,.30)
        with self.assertRaisesRegex(BreedingError,"duplicate"):
            bee_honey("b",[("c1",1,True),("c1",1,True)])

    def test_cross_module_or_wrong_coordinates_rejected(self):
        incompatible=plant("other",19.2,.85,module="different")
        with self.assertRaisesRegex(BreedingError,"incompatible"):
            breed_genomes(self.p[0].genome,incompatible.genome,
                bee("m"),self.specs,random.Random(1),child_id="kid",target_tile="sol-imba")
        p=replace(self.p[0],x=.9)
        with self.assertRaisesRegex(BreedingError,"coordinates must equal"):
            pollinate(DAY,5,[bee("b")],[p,self.p[1]],self.specs)

    def test_invalid_gene_schema_and_bee_rates_rejected(self):
        with self.assertRaisesRegex(BreedingError,"drop rate capped"):
            bee("bad",optional_gene_drop_rate=.2)
        with self.assertRaisesRegex(BreedingError,"required financial/safety"):
            bad=[s for s in self.specs if s.gene_id!="protective_stop"]
            spec_catalog(bad)
        with self.assertRaisesRegex(BreedingError,"duplicate gene"):
            spec_catalog(self.specs + [self.specs[0]])
        with self.assertRaisesRegex(BreedingError,"out of bounds"):
            validate_genome(gene("out",sens=100),self.specs)


if __name__=="__main__":
    unittest.main()
