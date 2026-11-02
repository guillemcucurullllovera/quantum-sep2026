#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the framework."""

from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream_tests import TrainingPipelineTestCase

from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.evaluation.efficient_depth_one import EfficientDepthOneEvaluator
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.evaluation.mps_evaluator import MPSEvaluator

from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.training.optimized_parameter_loader import OptimizedParametersLoader
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.training.parameter_scanner import DepthOneScanTrainer
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.training.random_point import RandomPoint
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.training.recursion import RecursionTrainer
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.training.reweighting import ReweightingTrainer
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.training.scipy_trainer import ScipyTrainer
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.training.tqa_trainer import TQATrainer
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.training.transition_states import TransitionStatesTrainer


class TestTrainingFramework(TrainingPipelineTestCase):
    """Generic tests of the framework."""

    def test_minimization_property(self):
        """Test the minimization property of the different trainers."""

        # Tests for trainers where minimization is not defined
        for cls in [RandomPoint, OptimizedParametersLoader]:
            trainer = cls()

            # pylint: disable=pointless-statement
            with self.assertRaises(ValueError):
                trainer.minimization

        for val in [True, False]:
            # Test for direct trainers.

            for cls in [DepthOneScanTrainer, ScipyTrainer, TQATrainer]:
                trainer = cls(MPSEvaluator(), energy_minimization=val)
                self.assertEqual(trainer.minimization, val)

            # Tests for trainers with sub-trainers
            sub_trainer = ScipyTrainer(EfficientDepthOneEvaluator(), energy_minimization=val)

            for cls in [ReweightingTrainer, RecursionTrainer, TransitionStatesTrainer]:
                trainer = cls(sub_trainer)
                self.assertEqual(trainer.minimization, val)
