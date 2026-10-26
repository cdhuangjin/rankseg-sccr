"""Portable full inference entry point; no downloads or overwriting outputs."""
import argparse,csv,hashlib,importlib.util,io,json,platform,tarfile,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--weights',required=True,type=Path)
    parser.add_argument('--official-code',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import torchvision
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large,LRASPP_MobileNet_V3_Large_Weights
    from sccr import coordinate_all,coordinate_pruned,objective
    from metrics import confusion,scores
    from io_utils import sha256,archive_bytes,write_json
    root=Path(__file__).resolve().parent
    hashes=json.loads((root/'input_checksums.json').read_text())
    cfg=json.loads((root/'protocol.json').read_text())
    if args.output.exists():raise FileExistsError('Use a new output directory; existing runs are preserved.')
    for f,k in [(args.archive,'archive_sha256'),(args.weights,'weights_sha256'),(args.official_code,'official_code_sha256')]:
        assert f.is_file(),f
        assert not f.resolve().is_relative_to(root),'Keep datasets/weights/upstream sources outside this minimal repository.'
        assert sha256(f)==hashes[k],f'Input checksum mismatch: {f.name}'
    spec=importlib.util.spec_from_file_location('official_rankseg_rma',args.official_code)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    assert official.TRUNCATE_PROB==cfg['pruning_probability']==0.1
    assert cfg['passes']==1 and cfg['order']=='row_major'
    torch.set_num_threads(cfg['threads']);torch.set_num_interop_threads(1)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    k=torch.randn(8,8);kernel=hashlib.sha256((k@k.T).numpy().tobytes()).hexdigest()
    assert kernel=='6ff91d277b2adf005c2fb882dc849140b82e3bc5c82e1200ccb985d9b4878e58','Environment witness differs; do not silently treat as an exact replication.'
    model=lraspp_mobilenet_v3_large(weights=None,weights_backbone=None)
    model.load_state_dict(torch.load(args.weights,weights_only=True,map_location='cpu'),strict=True)
    model.eval().requires_grad_(False)
    transform=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
    expected=json.loads((root/'results/manifest.json').read_text());methods=cfg['methods']
    rows=[];cms={m:[] for m in methods};diag=[]
    args.output.mkdir(parents=True);(args.output/'predictions').mkdir()
    write_json(args.output/'runtime.json',{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'kernel_sha256':kernel,'input_hashes':hashes,'config':cfg,'release_sources':{f.name:sha256(f) for f in root.glob('*.py')}})
    with tarfile.open(args.archive,'r:') as archive:
        raw=archive_bytes(archive,'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
        assert hashlib.sha256(raw).hexdigest()==hashes['split_sha256']
        ids=raw.decode().split()[100:180]
        assert ids==[r['id'] for r in expected]==(root/'confirm_b_ids.txt').read_text().split()
        for idx,(image_id,record) in enumerate(zip(ids,expected)):
            ib=archive_bytes(archive,f'VOCdevkit/VOC2012/JPEGImages/{image_id}.jpg')
            mb=archive_bytes(archive,f'VOCdevkit/VOC2012/SegmentationClass/{image_id}.png')
            assert hashlib.sha256(ib).hexdigest()==record['image_sha256']
            assert hashlib.sha256(mb).hexdigest()==record['mask_sha256']
            image=Image.open(io.BytesIO(ib)).convert('RGB');times={}
            with torch.inference_mode():
                logits=model(transform(image).unsqueeze(0))['out']
                p=F.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False).softmax(1)
                assert torch.isfinite(p).all()
                tick=time.perf_counter();argmax=p.argmax(1)[0].numpy().astype(np.uint8);times['argmax']=time.perf_counter()-tick
                tick=time.perf_counter();base=official.rankseg_rma(p.clone(),opt_metric='dice')[0].numpy();times['dice_rma']=time.perf_counter()-tick
                candidates=official.rankseg_rma(p.clone(),opt_metric='dice',allow_overlap=True)[0]
                overlap=(candidates.sum(0)>1).numpy();maximum=p[0].flatten(1).max(1).values.numpy();allowed=maximum>0.1
            probability=p[0].numpy()
            tick=time.perf_counter();all_pred,all_stats=coordinate_all(probability,base,overlap,cfg['move_tolerance']);times['coordinate_all']=time.perf_counter()-tick
            tick=time.perf_counter();pruned,pruned_stats=coordinate_pruned(probability,base,overlap,allowed,cfg['move_tolerance']);times['coordinate_pruned']=time.perf_counter()-tick
            preds=dict(zip(methods,[argmax,base,all_pred,pruned]))
            for pred in [all_pred,pruned]:assert np.array_equal(pred[~overlap],base[~overlap])
            assert allowed[pruned[overlap]].all()
            # Decode ground truth only after every prediction has been generated.
            gt=np.array(Image.open(io.BytesIO(mb)))
            arrays={**preds,'overlap':overlap,'allowed':allowed,'maximum_probability':maximum}
            for m,pred in preds.items():
                cm=confusion(pred,gt);cms[m].append(cm);arrays[m+'_confusion']=cm
                g=cm.sum(1);n=cm.sum(0)
                rows.append({'id':image_id,'method':m,**scores(cm),'stage_seconds':times[m],'surrogate_sum':objective(probability,pred),'predicted_classes':int((n>0).sum()),'false_only_classes':int(((g==0)&(n>0)).sum()),'false_only_pixels':int(n[g==0].sum())})
            np.savez_compressed(args.output/'predictions'/f'{image_id}.npz',**arrays)
            diag.append({'id':image_id,'coordinate_all':all_stats,'coordinate_pruned':pruned_stats})
            print(f'[{idx+1}/80] {image_id}',flush=True)
    with (args.output/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(args.output/'confusion_matrices.npz',**{m:np.stack(v) for m,v in cms.items()})
    write_json(args.output/'manifest.json',expected);write_json(args.output/'diagnostics.json',diag)
    write_json(args.output/'status.json',{'state':'complete','images':80})

if __name__=='__main__':main()
