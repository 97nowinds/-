"""Project extension of ByteTrack: evidence-gated Lost -> low-score recovery.

This is not the original ByteTrack algorithm or a part-aware ReID model.
Only current detector boxes are published; no predicted/held box is emitted.
"""
import time
import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching


class GuardedByteTracker(BYTETracker):
    def __init__(self, args, embedder):
        if not hasattr(BYTETracker, '_second_association'):
            raise RuntimeError('Guarded recovery requires the inspected ByteTrack stage API')
        super().__init__(args)
        self.embedder = embedder
        self.references = {}
        self.reference_gray = {}
        self.partial_tracks = {}
        self.evidence = {}
        self.audit = []
        self.budget_exhausted = False
        self.confirmed_current = set()
        self.remaining_pair_budget = 8

    def update(self, results, img=None, feats=None, observed_at=None, **kwargs):
        self.observed_at = time.monotonic() if observed_at is None else float(observed_at)
        self.image = img
        self.current_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.audit = []
        self.budget_exhausted = False
        self.confirmed_current = set()
        self.remaining_pair_budget = 8
        output = super().update(results, img, feats, **kwargs)
        # Store real detections, including evidence-confirmed partial pixels.
        # These short-lived visual references are separate from the clean
        # full-body identity gallery. Never store Kalman prediction as pixels.
        for row in output:
            tid=int(row[4])
            detection_index=int(row[-1])
            if detection_index<0 or detection_index>=len(results):
                continue
            actual=np.asarray(results.xyxy[detection_index])
            anchor=self.partial_tracks.get(tid)
            if anchor is not None:
                width,height=actual[2]-actual[0],actual[3]-actual[1]
                if height>=anchor[1]*.90 and width*height>=anchor[0]*anchor[1]*.85 and width/max(height,1)<=.60 and row[5]>=max(.60,self.args.track_high_thresh):
                    self.partial_tracks.pop(tid,None)
                elif tid not in self.confirmed_current:
                    continue
            if row[5] < self.args.track_high_thresh and tid not in self.confirmed_current:
                continue
            # The Kalman-smoothed box may shift/cut off the visible head.
            # Reference pixels must use the detection that actually produced
            # this observation, not the filtered tracker rectangle.
            detection_index = int(row[-1])
            if detection_index < 0 or detection_index >= len(results):
                continue
            x, y, r, b = np.round(results.xyxy[detection_index]).astype(int)
            h, w = img.shape[:2]
            x, y, r, b = max(0,x), max(0,y), min(w,r), min(h,b)
            if r-x >= 12 and b-y >= (24 if tid in self.confirmed_current else 32):
                self.references[int(row[4])] = (self.observed_at, (x,y,r,b), img[y:b,x:r].copy())
                self.reference_gray[int(row[4])] = self.current_gray
        alive = {t.track_id for t in self.tracked_stracks+self.lost_stracks}
        for tid in list(self.partial_tracks):
            if tid not in alive:
                self.partial_tracks.pop(tid,None)
        for tid in list(self.references):
            if tid not in alive or self.observed_at-self.references[tid][0] > self.args.lost_low_max_seconds:
                self.references.pop(tid, None)
                self.reference_gray.pop(tid, None)
                self.evidence.pop(tid, None)
        return output

    def _patches(self, track, detection):
        reference = self.references.get(track.track_id)
        if reference is None:
            return None, 'no_reference'
        stamp, (x,y,r,b), previous = reference
        age = self.observed_at-stamp
        if age <= 0 or age > self.args.lost_low_max_seconds:
            return None, 'stale_reference'
        xx, yy, rr, bb = np.round(detection.xyxy).astype(int)
        # Compare the same image region within the last actual ROI, suitable
        # only for a short stationary/slow partial reappearance. No full-body
        # vector is mislabelled as an upper-body vector.
        left, top, right, bottom = max(x,xx), max(y,yy), min(r,rr), min(b,bb)
        width, height = right-left, bottom-top
        coverage = max(0,width)*max(0,height)/max((rr-xx)*(bb-yy),1)
        scale = max(r-x,b-y,1)
        center = np.hypot((xx+rr-x-r)/2,(yy+bb-y-b)/2)/scale
        if track.track_id in self.partial_tracks:
            # The same visible head may now belong to a taller detector box.
            # Require shared visible pixels and head alignment, not body IoU.
            coverage=max(0,width)*max(0,height)/max(min((r-x)*(b-y),(rr-xx)*(bb-yy)),1)
            center=max(abs((xx+rr-x-r)/2)/max(r-x,rr-xx,1),abs(yy-y)/max(b-y,bb-yy,1))
            if abs(yy-y)>max(8,(b-y)*.20):
                return None,'head_alignment'
        minimum_height=24 if track.track_id in self.partial_tracks and track.state==TrackState.Tracked else 32
        if width < 12 or height < minimum_height or coverage < .70 or center > .45:
            return None, 'geometry'
        old_patch = previous[top-y:bottom-y,left-x:right-x]
        current_patch = self.image[top:bottom,left:right]
        if current_patch.shape != old_patch.shape:
            return None, 'invalid_patch'
        return (old_patch,current_patch,(left,top,right,bottom)),None

    def _appearance(self, track, detection):
        self.last_visual = {}
        patches,reason=self._patches(track,detection)
        if reason is not None:
            return None,reason
        old_patch,current_patch,region=patches
        if self.budget_exhausted:
            return None, 'comparison_budget'
        first,second=self.pair_features[(track.track_id,int(detection.idx))]
        cosine = None if first is None or second is None else float(np.dot(first,second))
        self.last_visual['appearance_similarity'] = cosine
        # A colour check complements the embedding; neither is an identity
        # guarantee when two people look alike or the ROI contains background.
        colour_delta = float(np.mean(np.abs(old_patch.astype(np.float32).mean((0,1))-current_patch.astype(np.float32).mean((0,1)))))/255
        if colour_delta > .18:
            return cosine, 'colour_change'
        if cosine is None:
            # A sub-32px head is below OSNet's crop admission size. Do not pad,
            # invent an embedding, or reacquire identity with it. Strong real
            # texture can only CONTINUE an adjacent confirmed observation.
            stamp=self.references[track.track_id][0]
            if track.state==TrackState.Tracked and track.track_id in self.partial_tracks and 0<self.observed_at-stamp<=.12:
                flow=self._flow(track.track_id,region,head_only=True)
                self.last_visual.update(flow)
                if flow.get('flow_verified') and flow.get('flow_median_ncc',0)>=.95:
                    self.last_visual['texture_only_continuity']=True
                    return .93,None
            return None,'invalid_feature'
        # Full-body OSNet is unreliable on a tiny head/shoulder patch. An
        # additional forward/backward LK correspondence can verify the SAME
        # visible texture over this short gap. This cannot bridge full occlusion.
        if getattr(self.args,'lost_low_use_flow',False) and cosine >= .60:
            flow = self._flow(track.track_id,region,head_only=True)
            if not flow.get('flow_verified',False):
                flow = self._flow(track.track_id,region)
            self.last_visual.update(flow)
            if flow.get('flow_verified',False):
                return max(cosine,.93),None
        # Turning a head changes its texture without changing the person. A
        # fresh ALREADY Tracked partial ROI can continue on appearance AND
        # tight actual-box motion. This cannot reassign a Lost identity.
        if track.state==TrackState.Tracked and track.track_id in self.partial_tracks and cosine>=.75:
            stamp,old_box,_=self.references[track.track_id]
            previous=np.asarray(old_box,dtype=float)
            current=np.asarray(detection.xyxy,dtype=float)
            intersection=np.prod(np.maximum(0,np.minimum(previous[2:],current[2:])-np.maximum(previous[:2],current[:2])))
            union=np.prod(previous[2:]-previous[:2])+np.prod(current[2:]-current[:2])-intersection
            iou=float(intersection/max(union,1))
            displacement=np.linalg.norm((previous[:2]+previous[2:]-current[:2]-current[2:])/2)/max(*(previous[2:]-previous[:2]),1)
            self.last_visual.update({'visible_box_iou':iou,'visible_center_step':float(displacement)})
            if 0<self.observed_at-stamp<=.12 and iou>=.70 and displacement<=.15:
                self.last_visual['pose_continuity_verified']=True
                return max(cosine,.90),None
        return cosine, None

    def _flow(self, tid, region, head_only=False):
        previous = self.reference_gray[tid]
        if previous.shape != self.current_gray.shape:
            return {'flow_verified':False}
        l,t,r,b = region
        b=min(b,t+(32 if head_only else 64))
        mask = np.zeros_like(previous)
        mask[t+2:b-2,l+2:r-2] = 255
        # Seed the currently VISIBLE patch. Seeding the old body instead makes
        # newly occluded corners dominate the denominator and rejects heads.
        points = cv2.goodFeaturesToTrack(self.current_gray,maxCorners=40,qualityLevel=.03,minDistance=3,mask=mask,blockSize=3)
        if points is None or len(points)<8:
            return {'flow_verified':False,'flow_points':0 if points is None else len(points)}
        params=dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.01))
        forward,status,error = cv2.calcOpticalFlowPyrLK(self.current_gray,previous,points,None,**params)
        if forward is None:
            return {'flow_verified':False}
        backward,back_status,_ = cv2.calcOpticalFlowPyrLK(previous,self.current_gray,forward,None,**params)
        if backward is None:
            return {'flow_verified':False}
        q,p,back = points[:,0],forward[:,0],backward[:,0]
        valid = (status[:,0]==1)&(back_status[:,0]==1)&(np.linalg.norm(q-back,axis=1)<.8)&(error[:,0]<12)
        old_l,old_t,old_r,old_b=self.references[tid][1]
        valid &= (p[:,0]>=old_l)&(p[:,0]<old_r)&(p[:,1]>=old_t)&(p[:,1]<old_b)
        count=int(valid.sum());ratio=count/len(points)
        fresh=0<self.observed_at-self.references[tid][0]<=.12
        active=any(track.track_id==tid and track.state==TrackState.Tracked for track in self.tracked_stracks)
        # A 32px visible head offers fewer corners than a body ROI. Eight
        # verified textured points may CONTINUE a fresh confirmed head; Lost
        # recovery and larger ROIs still require at least twelve.
        minimum=8 if head_only and fresh and active and tid in self.partial_tracks else 12
        # Occluded points are expected to disappear. Require a substantial
        # surviving set plus strong consensus, rather than 85% of ALL corners
        # (which would reject the intended partial-visibility case).
        if count<minimum or ratio<.40:
            return {'flow_verified':False,'flow_points':count,'flow_ratio':ratio}
        _,inliers = cv2.estimateAffinePartial2D(p[valid],q[valid],method=cv2.RANSAC,ransacReprojThreshold=1.5)
        consensus=float(inliers.mean()) if inliers is not None else 0.
        spread=np.ptp(p[valid],axis=0)
        median_error=float(np.median(error[:,0][valid]))
        correlations=[]
        for source,destination in zip(p[valid],q[valid]):
            old=cv2.getRectSubPix(previous,(7,7),tuple(map(float,source))).astype(float).reshape(-1)
            new=cv2.getRectSubPix(self.current_gray,(7,7),tuple(map(float,destination))).astype(float).reshape(-1)
            old-=old.mean();new-=new.mean()
            norm=np.linalg.norm(old)*np.linalg.norm(new)
            correlations.append(float(np.dot(old,new)/norm) if norm>1e-6 else -1.)
        median_ncc=float(np.median(correlations))
        # NCC verifies texture after point displacement and tolerates global
        # brightness/compression changes; flat patches cannot pass it.
        verified=count>=12 and consensus>=.90 and median_ncc>=.85 and np.mean(np.asarray(correlations)>=.70)>=.80 and spread[0]>=(r-l)*.25 and spread[1]>=(b-t)*.25
        # A newly occluding body introduces independently moving points. For
        # an ALREADY confirmed, still Tracked partial person, a substantial
        # RANSAC subset can establish continuity across adjacent observations.
        # This relaxed motion consensus never establishes/reacquires identity.
        continuation=False
        inlier_count=0
        inlier_ncc=None
        texture_count=0
        if inliers is not None and fresh and active and tid in self.partial_tracks:
            subset=inliers[:,0].astype(bool)
            inlier_count=int(subset.sum())
            selected=np.asarray(correlations)[subset]
            if inlier_count>=minimum and inlier_count/len(points)>=.40:
                inlier_ncc=float(np.median(selected))
                textured=selected>=.80
                texture_count=int(textured.sum())
                if texture_count>=minimum:
                    subset_spread=np.ptp(p[valid][subset][textured],axis=0)
                    continuation=consensus>=.65 and inlier_ncc>=.90 and texture_count/len(points)>=.40 and subset_spread[0]>=(r-l)*.25 and subset_spread[1]>=(b-t)*.25
        return {'flow_verified':bool(verified or continuation),'flow_strict_verified':bool(verified),
            'flow_continuation_verified':bool(continuation),'flow_inlier_points':inlier_count,
            'flow_inlier_ncc':inlier_ncc,'flow_texture_points':texture_count,'flow_points':count,'flow_ratio':ratio,'flow_consensus':consensus,'flow_median_error':median_error,'flow_median_ncc':median_ncc}

    @staticmethod
    def _same_visible_hypothesis(first,second):
        a,b=np.asarray(first.xyxy),np.asarray(second.xyxy)
        area_a=np.prod(a[2:]-a[:2]);area_b=np.prod(b[2:]-b[:2])
        intersection=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
        return intersection/max(min(area_a,area_b),1)>=.90 and abs(a[1]-b[1])<=4 and abs((a[0]+a[2]-b[0]-b[2])/2)<=min(a[2]-a[0],b[2]-b[0])*.20

    def _first_association(self, pool, detections, activated, refind):
        # Normal high-score association must not silently assign a recovered
        # partial track to a passing full-body box based on IoU alone.
        normal_indices=[i for i,t in enumerate(pool) if t.track_id not in self.partial_tracks]
        normal=[pool[i] for i in normal_indices]
        if normal and detections:
            missing,unused=super()._first_association(normal,detections,activated,refind)
            unmatched=[normal_indices[i] for i in missing]
        else:
            unmatched=normal_indices
            unused=list(range(len(detections)))
        partial_indices=[i for i,t in enumerate(pool) if t.track_id in self.partial_tracks]
        partial=[pool[i] for i in partial_indices]
        accepted=self._guarded_association(partial,[detections[i] for i in unused],activated,refind,'high')
        unmatched.extend(i for i in partial_indices if pool[i].track_id not in self.confirmed_current)
        return unmatched,[idx for j,idx in enumerate(unused) if j not in accepted and j not in self.reserved_high]

    def _second_association(self, pool, unmatched, low, activated, refind, lost):
        tracked = [pool[i] for i in unmatched if pool[i].state == TrackState.Tracked and pool[i].track_id not in self.partial_tracks]
        if tracked and low:
            matches, missing, unused = matching.linear_assignment(matching.iou_distance(tracked,low),thresh=.5)
            self._apply_matches(matches,tracked,low,activated,refind)
        else:
            missing, unused = list(range(len(tracked))), list(range(len(low)))
        # Only tracks that were already Lost entering this stage are eligible.
        # A low box already consumed by the normal stage can never be reused.
        candidates = [pool[i] for i in unmatched if pool[i].state == TrackState.Lost or pool[i].track_id in self.partial_tracks]
        remaining = [low[i] for i in unused]
        self._guarded_association(candidates,remaining,activated,refind,'low')
        for i in missing:
            track = tracked[i]
            track.mark_lost()
            lost.append(track)
        for track in candidates:
            if track.state==TrackState.Tracked and track.track_id not in self.confirmed_current:
                track.mark_lost()
                lost.append(track)

    def _guarded_association(self, candidates, remaining, activated, refind, stage):
        self.reserved_high=set()
        matrix = np.full((len(candidates),len(remaining)),-1.,dtype=float)
        visuals = {}
        valid_pairs=[]
        for track in candidates:
            for detection in remaining:
                patches,reason=self._patches(track,detection)
                if reason is None:
                    valid_pairs.append((track.track_id,int(detection.idx),patches[:2]))
        self.budget_exhausted=len(valid_pairs)>self.remaining_pair_budget
        self.pair_features={}
        if not self.budget_exhausted and valid_pairs:
            self.remaining_pair_budget-=len(valid_pairs)
            crops=[crop for _,_,pair in valid_pairs for crop in pair]
            extract_crops=getattr(self.embedder,'extract_crops',None)
            features=extract_crops(crops) if callable(extract_crops) else [self.embedder.extract(crop,(0,0,crop.shape[1],crop.shape[0])) for crop in crops]
            for index,(tid,idx,_) in enumerate(valid_pairs):
                self.pair_features[(tid,idx)]=(features[index*2],features[index*2+1])
        for i, track in enumerate(candidates):
            for j, detection in enumerate(remaining):
                similarity, reason = self._appearance(track,detection)
                visuals[(i,j)] = dict(self.last_visual)
                self.audit.append({'track_id':int(track.track_id),'detection_index':int(detection.idx),
                    'stage':stage,'association_score':similarity,'reason':reason or 'candidate','observed_at':self.observed_at,**self.last_visual})
                if reason is None:
                    matrix[i,j] = similarity
        accepted = set()
        for i, track in enumerate(candidates):
            if self.budget_exhausted:
                self.evidence.pop(track.track_id,None)
                continue
            if not len(remaining):
                break
            j = int(np.argmax(matrix[i]))
            value = matrix[i,j]
            competitors = [matrix[i,k] for k in range(len(remaining)) if k != j and not (
                visuals[(i,j)].get('flow_verified') and visuals[(i,k)].get('flow_verified')
                and self._same_visible_hypothesis(remaining[j],remaining[k]))]+[matrix[k,j] for k in range(len(candidates)) if k != i]
            margin = value-max(competitors) if competitors else 1.
            if value < self.args.lost_low_min_similarity or margin < self.args.lost_low_margin or j in accepted:
                # No geometric candidate in this score band must not erase
                # evidence accumulated in the other band during the same frame.
                if any(matrix[i]>=0):
                    self.evidence.pop(track.track_id,None)
                continue
            box = np.asarray(remaining[j].xyxy)
            old = self.evidence.get(track.track_id)
            continuous = old is not None and 0 < self.observed_at-old['last'] <= .20 and np.linalg.norm(box[:2]-old['box'][:2]) <= max(box[2]-box[0],box[3]-box[1])*.5
            count = old['count']+1 if continuous else 1
            first = old['first'] if continuous else self.observed_at
            self.evidence[track.track_id] = {'last':self.observed_at,'first':first,'count':count,'box':box.copy()}
            if stage=='high':
                # Three-observation Lost confirmation must finish before this
                # same high detection is used to create a competing new ID.
                self.reserved_high.add(j)
            continuing=track.state==TrackState.Tracked and track.track_id in self.partial_tracks
            if continuing or (count >= self.args.lost_low_confirmations and self.observed_at-first >= self.args.lost_low_min_duration):
                previous_box=self.references[track.track_id][1]
                self.partial_tracks.setdefault(track.track_id,(previous_box[2]-previous_box[0],previous_box[3]-previous_box[1]))
                self._apply_match(track,remaining[j],activated,refind)
                # Reinitialize shape from the real partial observation; the old
                # full-body Kalman shape would otherwise immediately lose IoU.
                track.mean,track.covariance = track.kalman_filter.initiate(track.convert_coords(remaining[j].tlwh))
                accepted.add(j)
                self.confirmed_current.add(track.track_id)
                self.evidence.pop(track.track_id,None)
                self.audit.append({'track_id':int(track.track_id),'detection_index':int(remaining[j].idx),
                    'stage':stage,'association_score':float(value),'reason':'continued' if continuing else 'recovered','observed_at':self.observed_at,**visuals[(i,j)]})
        return accepted

    def reset(self):
        super().reset()
        self.references.clear()
        self.reference_gray.clear()
        self.partial_tracks.clear()
        self.evidence.clear()
        self.audit.clear()
        self.confirmed_current.clear()
