from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import SingleStageDetector, TwoStageDetector
import torch


@DETECTORS.register_module()
class SingleStageDetBase(SingleStageDetector):
    def forward_train_w_feat(self,
                             feat,
                             img,
                             img_metas,
                             gt_bboxes,
                             gt_labels,
                             gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = feat
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses

    def set_detection_cfg(self, detection_cfg):
        self.bbox_head.test_cfg = detection_cfg

    def simple_test_w_feat(self, feat, img_metas, rescale=False):
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results


@DETECTORS.register_module()
class TwoStageDetBase(TwoStageDetector):
    def forward_train_w_feat(self,
                             feat,
                             img,
                             img_metas,
                             gt_bboxes,
                             gt_labels,
                             gt_bboxes_ignore=None,
                             gt_masks=None,
                             proposals=None,
                             **kwargs):
        x = feat

        losses = dict()
        gt_bboxes_np = gt_bboxes[0].detach().cpu().numpy() #debug
        print("gtbboxes shape: ", gt_bboxes_np.shape)#debug
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            #breakpoint()#debug
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
            #breakpoint()#debug
            print("START:img_metas[sample_idx]: ", img_metas[0]['sample_idx'])#debug

        else:
            proposal_list = proposals
        #breakpoint()#debug
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        print("DONE:img_metas[sample_idx]: ", img_metas[0]['sample_idx'])#debug

        return losses

    def set_detection_cfg(self, detection_cfg):
        self.roi_head.test_cfg = detection_cfg

    def simple_test_w_feat(self, feat, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = feat
        if proposals is None:
            #proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            # debug
            #split into 2 lists
            x1 = []
            x2 = []
            img_metas1 = []
            img_metas2 = []
            for tensor in x:
                split_x = torch.split(tensor, 5, dim=0)
                x1.append(split_x[0])
                x2.append(split_x[1])
                
            img_metas1 = img_metas[:5]
            img_metas2 = img_metas[5:]
            proposal_list1 = self.rpn_head.simple_test_rpn(x1, img_metas1)
            proposal_list2 = self.rpn_head.simple_test_rpn(x2, img_metas2)
            proposal_list = proposal_list1 + proposal_list2

        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
