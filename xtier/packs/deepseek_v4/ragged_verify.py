"""DSpark compact verification (SGLANG_RAGGED_VERIFY_MODE=compact) on DeepSeek V4. SGLang (since 27bc1a0) maps the
packed tokens to their requests itself, for the Engram hasher and the low-ratio attention, by the layout's qo_indptr
(which a graph replay refreshes); what is left here is the calibration budget (#13)."""


def forced_budget(result, self, frac):
    """After DSparkVerifyPlanner.set_forced_budget_frac: a calibration budget must bypass the verify-all shortcut of an
    uninitialized SPS table."""
    from sglang.srt.speculative.dspark_components.dspark_sps import is_uninitialized_sps_table

    if self._budget_planner is not None:
        self._is_verify_all = (self.is_compact_mode and frac is None
                               and is_uninitialized_sps_table(self._budget_planner.sps_table))
    return result
