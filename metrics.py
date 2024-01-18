import numpy as np
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Union, Dict
from abc import ABCMeta, abstractmethod

class BaseMetric(metaclass=ABCMeta):
    default_prefix: Optional[str] = "base_metric"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        collect_dir: Optional[str] = None,
    ) -> None:
        if collect_dir is not None and collect_device != "cpu":
            raise ValueError("`collec_dir` could only be configured when `collect_device='cpu'`")

        self.collect_device = collect_device
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix
        self.collect_dir = collect_dir

    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        pass

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        pass

    def evaluate(self, size: int) -> dict:
        metrics = self.compute_metrics(self.results)
        self.results.clear()
        return metrics


class IoUMetric(BaseMetric):
    def __init__(
        self,
        ignore_index: int = 255,
        iou_metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
        collect_device: str = "cpu",
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        classes: List[str] = ["background", "high_vegetation"],
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.classes = classes

    def process(self, input: Dict) -> None:
        num_classes = len(self.classes)

        pred_label = input["pred"].squeeze()

        label = input["gt"].squeeze()
        self.results.append(
            self.intersect_and_union(pred_label, label, num_classes, self.ignore_index)
        )

    def compute_metrics(self, results: list) -> Dict[str, float]:
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.classes

        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[key] = val
            else:
                metrics["m" + key] = val

        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        return metrics

    @staticmethod
    def intersect_and_union(
        pred_label: np.ndarray,
        label: np.ndarray,
        num_classes: int,
        ignore_index: int,
    ):
        mask = label != ignore_index
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect, _ = np.histogram(
            intersect.astype(float), bins=num_classes, range=(0, num_classes - 1)
        )
        area_pred_label, _ = np.histogram(
            pred_label.astype(float), bins=num_classes, range=(0, num_classes - 1)
        )

        area_label, _ = np.histogram(
            label.astype(float), bins=num_classes, range=(0, num_classes - 1)
        )
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(
        total_area_intersect: np.ndarray,
        total_area_union: np.ndarray,
        total_area_pred_label: np.ndarray,
        total_area_label: np.ndarray,
        metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
    ):
        def f_score(precision, recall, beta=1):
            score = (
                (1 + beta**2)
                * (precision * recall)
                / ((beta**2 * precision) + recall)
            )
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f"metrics {metrics} is not supported")

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({"aAcc": all_acc})
        for metric in metrics:
            if metric == "mIoU":
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics["IoU"] = iou
                ret_metrics["Acc"] = acc
            elif metric == "mDice":
                dice = (
                    2
                    * total_area_intersect
                    / (total_area_pred_label + total_area_label)
                )
                acc = total_area_intersect / total_area_label
                ret_metrics["Dice"] = dice
                ret_metrics["Acc"] = acc
            elif metric == "mFscore":
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = np.array(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
                )
                ret_metrics["Fscore"] = f_value
                ret_metrics["Precision"] = precision
                ret_metrics["Recall"] = recall

        ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}
        if nan_to_num is not None:
            ret_metrics = OrderedDict(
                {
                    metric: np.nan_to_num(metric_value, nan=nan_to_num)
                    for metric, metric_value in ret_metrics.items()
                }
            )
        return ret_metrics


class SMAPIoUMetric(IoUMetric):
    def compute_metrics(self, results: list) -> Dict[str, float]:
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.classes

        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[key] = val
            else:
                metrics["m" + key] = val

        for class_id, class_name in enumerate(class_names):
            for ret_metric, ret_metric_value in ret_metrics.items():
                if ret_metric == "aAcc":
                    continue
                metrics[f"{class_name}__{ret_metric}"] = np.round(ret_metric_value[class_id] * 100, 2)

        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        return metrics

    @staticmethod
    def total_area_to_metrics(
        total_area_intersect: np.ndarray,
        total_area_union: np.ndarray,
        total_area_pred_label: np.ndarray,
        total_area_label: np.ndarray,
        metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
    ):
        def f_score(precision, recall, beta=1):
            score = (
                (1 + beta**2)
                * (precision * recall)
                / ((beta**2 * precision) + recall)
            )
            return score

        if isinstance(metrics, str):
            metrics = [metrics]

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({"aAcc": all_acc})
        for metric in metrics:
            if metric == "mIoU":
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics["IoU"] = iou
                ret_metrics["Acc"] = acc
            elif metric == "mDice":
                dice = (
                    2
                    * total_area_intersect
                    / (total_area_pred_label + total_area_label)
                )
                acc = total_area_intersect / total_area_label
                ret_metrics["Dice"] = dice
                ret_metrics["Acc"] = acc
            elif metric == "mFscore":
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = np.array(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
                )
                ret_metrics["Fscore"] = f_value
                ret_metrics["Precision"] = precision
                ret_metrics["Recall"] = recall

        if nan_to_num is not None:
            ret_metrics = OrderedDict(
                {
                    metric: np.nan_to_num(metric_value, nan=nan_to_num)
                    for metric, metric_value in ret_metrics.items()
                }
            )
        return ret_metrics
