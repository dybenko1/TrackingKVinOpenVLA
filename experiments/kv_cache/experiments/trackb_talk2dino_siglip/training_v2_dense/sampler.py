"""Collision-free full-caption sampler and deterministic category selection."""
from __future__ import annotations
import random
from collections import defaultdict


class CaptionRoundSampler:
    def __init__(self, image_ids, batch_size=128, seed=123):
        self.image_ids = tuple(map(int, image_ids)); self.batch_size = int(batch_size); self.seed = int(seed); self.epoch = 0
        if not self.image_ids or self.batch_size < 1: raise ValueError("nonempty IDs and positive batch size required")
        self.groups = defaultdict(list)
        for i, iid in enumerate(self.image_ids): self.groups[iid].append(i)
    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __iter__(self):
        rng = random.Random((self.seed << 32) ^ self.epoch)
        groups = []
        for iid, rows0 in self.groups.items():
            rows = rows0.copy(); rng.shuffle(rows); groups.append((iid, rows))
        for r in range(max(map(lambda x: len(x[1]), groups))):
            rows = [rs[r] for _, rs in groups if r < len(rs)]; rng.shuffle(rows)
            for i in range(0, len(rows), self.batch_size): yield rows[i:i+self.batch_size]
    def __len__(self):
        return sum((sum(len(rs) > r for _, rs in self.groups.items()) + self.batch_size-1)//self.batch_size for r in range(max(map(len,self.groups.values()))))


def choose_categories(image_id, candidate_categories, area_by_category, category_frequency, epoch, visit, limit=4, seed=123):
    """Small-object first, inverse-sqrt frequency next, then deterministic fill."""
    cands = [int(x) for x in candidate_categories if float(area_by_category[int(x)]) > 0]
    if not cands: return []
    rng = random.Random((int(seed)<<40) ^ (int(epoch)<<24) ^ (int(image_id)<<4) ^ int(visit))
    selected=[]
    small=sorted(cands,key=lambda c:(float(area_by_category[c]),c))
    selected.append(small[0])
    remain=[c for c in cands if c not in selected]
    if remain and len(selected)<limit:
        weights=[1.0/(max(1,int(category_frequency[c]))**0.5) for c in remain]
        x=rng.random()*sum(weights); acc=0.0
        for c,w in zip(remain,weights):
            acc+=w
            if acc>=x: selected.append(c); break
    remain=[c for c in cands if c not in selected]
    rng.shuffle(remain); selected.extend(remain[:max(0,limit-len(selected))])
    return selected
