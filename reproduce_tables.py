"""Recompute reported metrics and paired intervals from saved numerical evidence."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from metrics import scores

def reproduce(folder,reference):
    rows=list(csv.DictReader((folder/'per_image.csv').open(encoding='utf-8-sig')))
    ids=[x['id'] for x in json.loads((folder/'manifest.json').read_text())]
    expected=json.loads((reference/'summary.json').read_text())
    methods=list(expected['methods'])
    assert len(ids)==80 and len(set(ids))==80 and len(rows)==320
    assert ids==(Path(__file__).parent/'confirm_b_ids.txt').read_text().split()
    keyed={(r['id'],r['method']):r for r in rows};assert len(keyed)==320
    metrics={};comp={}
    with np.load(folder/'confusion_matrices.npz') as saved:
        for m in methods:
            cm=saved[m];assert cm.shape==(80,21,21)
            rs=[]
            for image_id,c in zip(ids,cm):
                assert (c>=0).all() and np.equal(c,np.floor(c)).all()
                s=scores(c);g=c.sum(1);p=c.sum(0)
                s.update(false_only_classes=int(((g==0)&(p>0)).sum()),false_only_pixels=int(p[g==0].sum()))
                for k,v in s.items():assert np.isclose(v,float(keyed[image_id,m][k]),atol=1e-10,rtol=0),(image_id,m,k)
                rs.append(s)
            metrics[m]={k:float(np.mean([r[k] for r in rs])) for k in rs[0] if k!='false_only_pixels'}
            metrics[m]['false_only_pixels_total']=sum(r['false_only_pixels'] for r in rs)
            metrics[m]['dataset_miou']=scores(cm.sum(0))['miou_image_union_present']
            for k,v in metrics[m].items():assert np.isclose(v,expected['methods'][m][k],atol=1e-8,rtol=0),(m,k,v)
    for name,comparison in expected['paired_comparisons'].items():
        treatment,control=name.split('_vs_');comp[name]={}
        for metric,target in comparison.items():
            delta=np.array([float(keyed[i,treatment][metric])-float(keyed[i,control][metric]) for i in ids])
            rng=np.random.default_rng(20261023)
            ci=np.percentile(rng.choice(delta,(5000,80),replace=True).mean(1),[2.5,97.5])
            assert np.isclose(delta.mean(),target['mean_delta'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(ci,target['ci95'],atol=1e-10,rtol=0)
            comp[name][metric]={'mean_delta':float(delta.mean()),'ci95':ci.tolist()}
    print(json.dumps({'status':'PASS','images':80,'checked_confusion_matrices':320,'metrics':metrics,'paired_comparisons':comp},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args()
    reproduce(args.results,Path(__file__).parent/'results')
