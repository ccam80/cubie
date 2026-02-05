"""Minimal timing loop test to diagnose infinite hang conditions.

This script replicates only the timing/save boundary logic from ode_loop.py
to trace the exact failure chain when using inexact float32 save intervals.

Key variables from ode_loop:
- t: float64 - high precision time accumulator
- t_prec: float32 - precision-cast time for step calculations
- t_end: float32 - end time
- next_save: float32 - next scheduled save time
- save_every: float32 - save interval
- dt_raw: float32 - raw timestep from controller
- dt_eff: float32 - effective timestep (possibly clipped to hit save boundary)
"""

import numpy as np
from numpy import float32, float64, int32


def get_float32_ulp(value: float) -> float:
    """Get the unit in last place (ULP) for a float32 value."""
    f32_val = float32(value)
    # ULP = value * 2^-23 (for normalized floats)
    return float(abs(f32_val)) * (2.0 ** -23)


def analyze_float32(value: float, name: str = "value") -> dict:
    """Analyze float32 representation details."""
    f64_val = float64(value)
    f32_val = float32(value)
    f32_as_f64 = float64(f32_val)

    return {
        "name": name,
        "f64": f64_val,
        "f32": f32_val,
        "f32_as_f64": f32_as_f64,
        "error": float(f64_val - f32_as_f64),
        "ulp_at_scale": get_float32_ulp(f32_val),
        "is_exact": f64_val == f32_as_f64,
    }


def print_float_analysis(info: dict):
    """Pretty print float analysis."""
    print(f"  {info['name']}:")
    print(f"    f64 value:     {info['f64']:.20g}")
    print(f"    f32 value:     {float(info['f32']):.20g}")
    print(f"    f32→f64:       {info['f32_as_f64']:.20g}")
    print(f"    error (f64-f32): {info['error']:.6e}")
    print(f"    f32 ULP at scale: {info['ulp_at_scale']:.6e}")
    print(f"    is exact f32:  {info['is_exact']}")


def simulate_timing_loop(
    t0: float,
    duration: float,
    settling_time: float,
    save_every_f64: float,
    dt_raw_val: float,
    max_iterations: int = 200,
    verbose: bool = True,
) -> dict:
    """Simulate the timing logic from ode_loop.

    This replicates the core timing mechanics without the actual ODE solving.

    Parameters
    ----------
    t0 : float
        Initial time
    duration : float
        Integration duration
    settling_time : float
        Settling time before saving starts
    save_every_f64 : float
        Save interval (will be cast to float32 as in actual code)
    dt_raw_val : float
        Raw timestep value (simulated controller output)
    max_iterations : int
        Safety limit to detect hangs
    verbose : bool
        Print detailed trace

    Returns
    -------
    dict
        Simulation results including iteration count, final state, etc.
    """
    # Cast to working precision (mimics ode_loop)
    precision = float32

    # Initialize timing variables exactly as in ode_loop lines 541-605
    t = float64(t0)
    t_prec = precision(t)
    t_end = precision(settling_time + t0 + duration)
    save_every = precision(save_every_f64)

    next_save = precision(settling_time + t0)
    dt_raw = precision(dt_raw_val)

    stagnant_counts = int32(0)
    save_idx = int32(0)

    # Skip initial save at t0 (assume settling_time=0)
    if settling_time == 0.0:
        next_save = precision(next_save + save_every)
        save_idx += int32(1)

    if verbose:
        print("=" * 80)
        print("TIMING LOOP SIMULATION")
        print("=" * 80)
        print("\nInitial configuration:")
        print_float_analysis(analyze_float32(save_every_f64, "save_every"))
        print_float_analysis(analyze_float32(float(t_end), "t_end"))
        print_float_analysis(analyze_float32(float(next_save), "next_save"))
        print(f"\n  dt_raw: {float(dt_raw):.10g}")
        print(f"  Expected saves: {int(duration / save_every_f64)}")
        print()

    results = {
        "iterations": 0,
        "saves": int(save_idx),
        "final_t": float(t),
        "final_t_prec": float(t_prec),
        "stagnant": False,
        "oscillation_detected": False,
        "history": [],
    }

    # Track for oscillation detection
    recent_t_values = []

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # ===== Lines 691-703: Check finish condition =====
        end_of_step = t_prec + dt_raw
        save_finished = bool(next_save > t_end)
        finished = save_finished

        if finished:
            if verbose:
                print(f"\n[Iter {iteration}] FINISHED: next_save ({float(next_save):.10g}) > t_end ({float(t_end):.10g})")
            results["finished_normally"] = True
            break

        # ===== Lines 719-747: Determine if save is due and compute dt_eff =====
        do_save = bool(end_of_step >= next_save) and not save_finished

        dt_eff = dt_raw
        if do_save:
            next_event = precision(min(float(t_end), float(next_save)))
            dt_eff = precision(next_event - t_prec)

        # ===== Lines 827-844: Time advancement =====
        t_proposal = t + float64(dt_eff)

        # Stagnation check (lines 832-837)
        if t_proposal == t:
            stagnant_counts += int32(1)
        else:
            stagnant_counts = int32(0)

        stagnant = bool(stagnant_counts >= int32(2))

        # Track state for this iteration
        step_info = {
            "iter": iteration,
            "t": float(t),
            "t_prec": float(t_prec),
            "t_proposal": float(t_proposal),
            "next_save": float(next_save),
            "dt_eff": float(dt_eff),
            "dt_eff_f64": float(float64(dt_eff)),
            "do_save": do_save,
            "stagnant_counts": int(stagnant_counts),
            "t_proposal_minus_t": float(t_proposal - t),
            "next_save_minus_t_prec": float(float64(next_save) - float64(t_prec)),
        }
        results["history"].append(step_info)

        if verbose and (iteration <= 10 or do_save or stagnant_counts > 0 or
                       float(next_save) - float(t_prec) < 1e-5):
            print(f"\n[Iter {iteration}]")
            print(f"  t (f64):        {float(t):.15g}")
            print(f"  t_prec (f32):   {float(t_prec):.15g}")
            print(f"  next_save:      {float(next_save):.15g}")
            print(f"  next_save - t_prec: {float(float64(next_save) - float64(t_prec)):.6e}")
            print(f"  dt_eff:         {float(dt_eff):.15g} (f64: {float(float64(dt_eff)):.6e})")
            print(f"  t_proposal:     {float(t_proposal):.15g}")
            print(f"  t_proposal - t: {float(t_proposal - t):.6e}")
            print(f"  do_save:        {do_save}")
            print(f"  stagnant_counts: {stagnant_counts}")

        if stagnant:
            if verbose:
                print(f"\n*** STAGNATION DETECTED at iteration {iteration} ***")
            results["stagnant"] = True
            break

        # Detect oscillation: t going backward
        if t_proposal < t:
            if verbose:
                print(f"\n*** BACKWARD TIME DETECTED: t_proposal ({float(t_proposal):.15g}) < t ({float(t):.15g}) ***")
                print(f"    dt_eff was: {float(dt_eff):.6e}")
            results["oscillation_detected"] = True

        # Check for oscillation pattern in recent values
        recent_t_values.append(float(t_prec))
        if len(recent_t_values) > 10:
            recent_t_values.pop(0)
            # Check if t_prec is oscillating between two values
            unique_recent = set(recent_t_values)
            if len(unique_recent) <= 2 and len(recent_t_values) >= 10:
                if verbose:
                    print(f"\n*** OSCILLATION PATTERN: t_prec stuck at {unique_recent} ***")
                results["oscillation_detected"] = True

        # Accept step (simulate always-accept for simplicity)
        accept = True
        if accept:
            t = t_proposal
            t_prec = precision(t)

        # ===== Lines 872-876: Save handling =====
        if do_save and accept:
            next_save = precision(next_save + save_every)
            save_idx += int32(1)
            if verbose:
                print(f"  >>> SAVE #{save_idx}: next_save now {float(next_save):.15g}")

    results["iterations"] = iteration
    results["saves"] = int(save_idx)
    results["final_t"] = float(t)
    results["final_t_prec"] = float(t_prec)

    if iteration >= max_iterations:
        results["timeout"] = True
        if verbose:
            print(f"\n*** TIMEOUT at {max_iterations} iterations ***")

    return results


def run_test_cases():
    """Run the problematic test cases identified in the issue."""
    print("\n" + "=" * 80)
    print("TEST CASE ANALYSIS")
    print("=" * 80)

    # Common parameters
    t0 = 0.0
    settling_time = 0.0
    dt_raw = 0.01  # Reasonable step size for demonstration

    test_cases = [
        # (description, save_every, duration, expected_behavior)
        ("Inexact f32, duration=8.0 (reportedly works)", 0.1, 8.0, "works"),
        ("Inexact f32, duration=8.05 (reportedly HANGS)", 0.1, 8.05, "hangs"),
        ("Exact f32 (0.125), duration=10.0 (works)", 0.125, 10.0, "works"),
        ("Inexact f32, duration=8.1 (test)", 0.1, 8.1, "unknown"),
        ("Inexact f32, duration=7.9 (test)", 0.1, 7.9, "unknown"),
    ]

    for desc, save_every, duration, expected in test_cases:
        print(f"\n{'=' * 80}")
        print(f"TEST: {desc}")
        print(f"  save_every={save_every}, duration={duration}")
        print(f"  Expected behavior: {expected}")
        print("=" * 80)

        result = simulate_timing_loop(
            t0=t0,
            duration=duration,
            settling_time=settling_time,
            save_every_f64=save_every,
            dt_raw_val=dt_raw,
            max_iterations=200,
            verbose=True,
        )

        print(f"\nRESULT:")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Saves: {result['saves']}")
        print(f"  Final t: {result['final_t']:.15g}")
        print(f"  Stagnant: {result['stagnant']}")
        print(f"  Oscillation: {result.get('oscillation_detected', False)}")
        print(f"  Timeout: {result.get('timeout', False)}")


def analyze_boundary_accumulation():
    """Analyze what happens as next_save accumulates with inexact f32."""
    print("\n" + "=" * 80)
    print("SAVE BOUNDARY ACCUMULATION ANALYSIS")
    print("=" * 80)

    save_every_f64 = 0.1
    save_every_f32 = float32(save_every_f64)

    print(f"\nsave_every as f64: {save_every_f64:.20g}")
    print(f"save_every as f32: {float(save_every_f32):.20g}")
    print(f"f32 representation error: {float(save_every_f32) - save_every_f64:.6e}")

    print(f"\nAccumulated next_save values (80 increments to reach t~8.0):")
    print(f"{'n':>3} | {'next_save (f32)':>20} | {'ideal (n*0.1)':>15} | {'error':>12} | {'f32 ULP':>12}")
    print("-" * 75)

    next_save = float32(0.0)
    for n in range(1, 85):
        next_save = float32(next_save + save_every_f32)
        ideal = n * 0.1
        error = float(next_save) - ideal
        ulp = get_float32_ulp(next_save)

        if n <= 5 or n >= 78:
            print(f"{n:3d} | {float(next_save):20.15g} | {ideal:15.10g} | {error:12.6e} | {ulp:12.6e}")

    # Focus on the boundary at n=80
    print(f"\n\nCritical boundary analysis at n=80 (t≈8.0):")
    next_save_80 = float32(0.0)
    for _ in range(80):
        next_save_80 = float32(next_save_80 + save_every_f32)

    t_end_8 = float32(8.0)
    t_end_805 = float32(8.05)

    print(f"  next_save after 80 increments: {float(next_save_80):.20g}")
    print(f"  t_end = 8.0:  {float(t_end_8):.20g}")
    print(f"  t_end = 8.05: {float(t_end_805):.20g}")
    print(f"  next_save > t_end(8.0):  {next_save_80 > t_end_8}")
    print(f"  next_save > t_end(8.05): {next_save_80 > t_end_805}")

    # If next_save > t_end(8.0), the loop would exit before attempting save at 8.0
    # If next_save <= t_end(8.05), the loop attempts to hit that save boundary

    print(f"\n  Difference next_save - t_end(8.0):  {float(next_save_80) - 8.0:.6e}")
    print(f"  Difference next_save - t_end(8.05): {float(next_save_80) - 8.05:.6e}")


def trace_critical_boundary():
    """Trace exactly what happens at the t≈8.0 boundary with duration=8.05."""
    print("\n" + "=" * 80)
    print("CRITICAL BOUNDARY TRACE (t approaching 8.0, duration=8.05)")
    print("=" * 80)

    precision = float32
    save_every = precision(0.1)
    t_end = precision(8.05)

    # Accumulate next_save to step 80
    next_save = precision(0.0)
    for _ in range(81):  # 0 + 81 increments = 81 saves (0 through 80*0.1)
        next_save = precision(next_save + save_every)

    print(f"\nAfter 81 save_every increments:")
    print(f"  next_save = {float(next_save):.20g}")
    print(f"  t_end     = {float(t_end):.20g}")
    print(f"  next_save > t_end: {next_save > t_end}")

    # Simulate t approaching this boundary
    print(f"\nSimulating t approaching next_save boundary:")

    # Start t just before the boundary
    t = float64(7.999)
    dt_raw = precision(0.01)

    for step in range(20):
        t_prec = precision(t)
        end_of_step = t_prec + dt_raw

        save_finished = bool(next_save > t_end)
        do_save = bool(end_of_step >= next_save) and not save_finished

        if do_save:
            next_event = precision(min(float(t_end), float(next_save)))
            dt_eff = precision(next_event - t_prec)
        else:
            dt_eff = dt_raw

        t_proposal = t + float64(dt_eff)

        # Key diagnostic values
        print(f"\n  Step {step}:")
        print(f"    t (f64):        {float(t):.15g}")
        print(f"    t_prec (f32):   {float(t_prec):.15g}")
        print(f"    next_save:      {float(next_save):.15g}")
        print(f"    dt_eff:         {float(dt_eff):.10g} ({float(dt_eff):.6e})")
        print(f"    t_proposal:     {float(t_proposal):.15g}")
        print(f"    do_save:        {do_save}")
        print(f"    dt_eff < 0:     {dt_eff < 0}")
        print(f"    t_proposal < t: {t_proposal < t}")

        if t_proposal < t:
            print(f"    *** BACKWARD TIME DETECTED ***")
            break

        if t_proposal == t:
            print(f"    *** STAGNATION DETECTED ***")
            break

        # Accept the step
        t = t_proposal

        if do_save:
            print(f"    >>> WOULD SAVE, advancing next_save")
            next_save = precision(next_save + save_every)
            print(f"    >>> new next_save: {float(next_save):.15g}")

        if save_finished or next_save > t_end:
            print(f"    >>> Loop would finish")
            break


def find_negative_dt_eff_scenario():
    """Search for conditions where dt_eff becomes negative."""
    print("\n" + "=" * 80)
    print("SEARCHING FOR NEGATIVE dt_eff SCENARIOS")
    print("=" * 80)

    precision = float32

    # The key insight: dt_eff = next_save - t_prec
    # dt_eff can be negative when t_prec > next_save
    # This happens when t (f64) rounds UP to a t_prec that exceeds next_save

    # For t_prec to EXCEED next_save, we need:
    # 1. t (f64) to round UP when cast to f32
    # 2. The rounded value to be greater than next_save

    # This can happen when next_save is between two f32 values,
    # and t is between next_save and the next higher f32 value

    print("\nLooking for t_prec > next_save scenarios...")

    for save_num in [10, 40, 70, 80]:
        save_every = precision(0.1)
        next_save = precision(0.0)
        for _ in range(save_num):
            next_save = precision(next_save + save_every)

        ns_val = float(next_save)
        next_f32 = float(np.nextafter(precision(ns_val), precision(np.inf)))

        print(f"\n  Save #{save_num}:")
        print(f"    next_save (f32):     {ns_val:.20g}")
        print(f"    next higher f32:     {next_f32:.20g}")

        # Check if t in (next_save, next_f32) rounds to next_f32
        # The rounding boundary between next_save and next_f32 is at their midpoint
        boundary = (ns_val + next_f32) / 2

        print(f"    rounding boundary:   {boundary:.20g}")

        # Test values just above boundary (should round to next_f32)
        for offset in [1e-12, 1e-10, 1e-8]:
            t = float64(boundary + offset)
            t_prec = precision(t)
            # dt_eff = next_save - t_prec
            dt_eff = precision(ns_val - float(t_prec))

            marker = ""
            if float(t_prec) > ns_val:
                if dt_eff < 0:
                    marker = " <-- NEGATIVE! t_prec > next_save"
                else:
                    marker = f" t_prec > next_save but dt_eff >= 0 (?)"
            print(f"    t=boundary+{offset:.0e}: t_prec={float(t_prec):.15g}, "
                  f"dt_eff={float(dt_eff):.6e}{marker}")


def detailed_f64_f32_rounding():
    """Show exactly how f64→f32 rounding affects time progression."""
    print("\n" + "=" * 80)
    print("FLOAT64 TO FLOAT32 ROUNDING EFFECTS")
    print("=" * 80)

    precision = float32

    # The critical observation: when t (f64) is slightly below a f32 value,
    # t_prec = float32(t) can round UP, potentially overshooting next_save

    save_every = precision(0.1)
    next_save = precision(0.0)
    for _ in range(80):  # Get to save #80 at ~8.0
        next_save = precision(next_save + save_every)

    print(f"\nnext_save at iteration 80: {float(next_save):.20g}")

    # Find consecutive f32 values around next_save
    ns_f32 = float(next_save)
    ns_prev = float(np.nextafter(precision(ns_f32), precision(-np.inf)))
    ns_next = float(np.nextafter(precision(ns_f32), precision(np.inf)))

    print(f"Previous f32: {ns_prev:.20g}")
    print(f"next_save:    {ns_f32:.20g}")
    print(f"Next f32:     {ns_next:.20g}")

    # Find the f64 rounding boundary
    boundary = (ns_prev + ns_f32) / 2
    print(f"\nRounding boundary (below this rounds to ns_prev, above to ns_f32):")
    print(f"  {boundary:.20g}")

    # What happens if t (f64) is right at the boundary?
    print(f"\nWhat happens at the rounding boundary:")
    for offset in [-1e-10, -1e-15, 0, 1e-15, 1e-10]:
        t = float64(boundary + offset)
        t_prec = precision(t)
        dt_eff = precision(float(next_save) - float(t_prec))
        print(f"  t = boundary + {offset:+.0e}: t_prec={float(t_prec):.15g}, dt_eff={float(dt_eff):.6e}")


def simulate_with_step_rejection():
    """Simulate what happens when steps can be rejected (dt_eff=0 case)."""
    print("\n" + "=" * 80)
    print("SIMULATION WITH dt_eff=0 (STEP REJECTION SCENARIO)")
    print("=" * 80)

    precision = float32
    save_every = precision(0.1)
    dt_raw = precision(0.01)

    # Start t slightly before a save boundary where dt_eff=0 can occur
    # From earlier output, this happens around t≈0.4 and t≈0.5
    t = float64(0.39)
    next_save = precision(0.4)  # After 4 increments

    print(f"\nStarting t={float(t)}, next_save={float(next_save)}")
    print(f"Simulating progression with step acceptance/rejection...")

    stagnant_counts = int32(0)

    for step in range(30):
        t_prec = precision(t)
        end_of_step = t_prec + dt_raw

        do_save = bool(end_of_step >= next_save)

        if do_save:
            dt_eff = precision(float(next_save) - float(t_prec))
        else:
            dt_eff = dt_raw

        t_proposal = t + float64(dt_eff)

        # Stagnation check
        if t_proposal == t:
            stagnant_counts += int32(1)
        else:
            stagnant_counts = int32(0)

        stagnant = bool(stagnant_counts >= int32(2))

        print(f"\n  Step {step}:")
        print(f"    t (f64):        {float(t):.15g}")
        print(f"    t_prec (f32):   {float(t_prec):.15g}")
        print(f"    next_save:      {float(next_save):.15g}")
        print(f"    dt_eff:         {float(dt_eff):.6e}")
        print(f"    t_proposal:     {float(t_proposal):.15g}")
        print(f"    t_proposal==t:  {t_proposal == t}")
        print(f"    stagnant_counts: {stagnant_counts}")

        if stagnant:
            print(f"\n    *** STAGNATION TRIGGERED ***")
            print(f"    This would set error status and exit the loop")
            break

        # In this simple sim, always accept
        t = t_proposal

        if do_save:
            next_save = precision(next_save + save_every)
            print(f"    >>> SAVE, next_save now {float(next_save):.15g}")

        if step > 20:
            print(f"    >>> Stopping simulation")
            break


def simulate_full_loop(duration: float, verbose_threshold: float = None):
    """Simulate the full timing loop from t=0 to detect hangs."""
    print(f"\n{'=' * 80}")
    print(f"FULL LOOP SIMULATION: duration={duration}")
    print("=" * 80)

    precision = float32

    t = float64(0.0)
    t_prec = precision(t)
    t_end = precision(duration)
    save_every = precision(0.1)
    dt_raw = precision(0.01)  # Typical step size

    next_save = precision(save_every)  # Start at first save (after initial save at t=0)
    save_idx = 1  # Already saved at t=0

    stagnant_counts = int32(0)
    backward_count = 0
    max_iterations = 100000

    print(f"\n  t_end = {float(t_end):.15g}")
    print(f"  save_every = {float(save_every):.15g}")

    for iteration in range(max_iterations):
        t_prec = precision(t)
        end_of_step = t_prec + dt_raw

        save_finished = bool(next_save > t_end)
        finished = save_finished

        if finished:
            print(f"\n[Iter {iteration}] FINISHED normally")
            print(f"  Total saves: {save_idx}")
            print(f"  Final t: {float(t):.15g}")
            print(f"  Backward movements: {backward_count}")
            return {"success": True, "iterations": iteration, "saves": save_idx,
                    "backward_count": backward_count}

        do_save = bool(end_of_step >= next_save) and not save_finished

        if do_save:
            next_event = precision(min(float(t_end), float(next_save)))
            dt_eff = precision(float(next_event) - float(t_prec))
        else:
            dt_eff = dt_raw

        t_proposal = t + float64(dt_eff)

        # Stagnation check
        if t_proposal == t:
            stagnant_counts += int32(1)
        else:
            stagnant_counts = int32(0)

        stagnant = bool(stagnant_counts >= int32(2))

        # Verbose output near interesting points
        is_interesting = (
            (verbose_threshold and float(t) >= verbose_threshold) or
            dt_eff < 0 or
            t_proposal < t or
            stagnant_counts > 0
        )

        if is_interesting:
            print(f"\n[Iter {iteration}] t={float(t):.15g}")
            print(f"  t_prec={float(t_prec):.15g}, next_save={float(next_save):.15g}")
            print(f"  dt_eff={float(dt_eff):.6e}, do_save={do_save}")
            print(f"  t_proposal={float(t_proposal):.15g}")
            print(f"  backward: {t_proposal < t}, stagnant_counts: {stagnant_counts}")

        if stagnant:
            print(f"\n*** STAGNATION at iter {iteration}, t={float(t):.15g} ***")
            return {"success": False, "reason": "stagnation", "iterations": iteration,
                    "saves": save_idx, "backward_count": backward_count}

        # Track backward movements
        if t_proposal < t:
            backward_count += 1
            if backward_count > 1000:
                print(f"\n*** EXCESSIVE BACKWARD MOVEMENTS: {backward_count} ***")
                print(f"  Likely in oscillation loop")
                return {"success": False, "reason": "oscillation", "iterations": iteration,
                        "saves": save_idx, "backward_count": backward_count}

        # Accept step
        accept = True
        t = t_proposal
        t_prec = precision(t)

        if do_save and accept:
            next_save = precision(next_save + save_every)
            save_idx += 1

    print(f"\n*** TIMEOUT at {max_iterations} iterations ***")
    return {"success": False, "reason": "timeout", "iterations": max_iterations,
            "saves": save_idx, "backward_count": backward_count}


def simulate_oscillation_failure():
    """Run full simulations to find hang conditions."""
    print("\n" + "=" * 80)
    print("SEARCHING FOR HANG CONDITIONS")
    print("=" * 80)

    test_durations = [8.0, 8.01, 8.02, 8.03, 8.04, 8.05, 8.1, 10.0]

    results = []
    for dur in test_durations:
        result = simulate_full_loop(dur, verbose_threshold=dur - 0.1)
        result["duration"] = dur
        results.append(result)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        status = "OK" if r["success"] else f"FAIL ({r.get('reason', 'unknown')})"
        print(f"  duration={r['duration']}: {status}, saves={r['saves']}, "
              f"iters={r['iterations']}, backward={r['backward_count']}")


def trace_ulp_scale_increments():
    """Trace what happens when dt_eff is at ULP scale."""
    print("\n" + "=" * 80)
    print("ULP-SCALE INCREMENT ANALYSIS")
    print("=" * 80)

    precision = float32

    # Build up next_save to save 80
    save_every = precision(0.1)
    next_save = precision(0.0)
    for _ in range(80):
        next_save = precision(next_save + save_every)

    t_end = precision(8.05)

    print(f"\nnext_save = {float(next_save):.20g}")
    print(f"t_end = {float(t_end):.15g}")

    # Get consecutive f32 values around next_save
    ns_val = float(next_save)
    ns_prev = float(np.nextafter(precision(ns_val), precision(-np.inf)))
    ns_next = float(np.nextafter(precision(ns_val), precision(np.inf)))

    print(f"\nConsecutive f32 values:")
    print(f"  ns_prev:    {ns_prev:.20g}")
    print(f"  next_save:  {ns_val:.20g}")
    print(f"  ns_next:    {ns_next:.20g}")
    print(f"  ULP size:   {ns_val - ns_prev:.6e}")

    dt_raw = precision(0.01)

    # Trace what happens when t_prec is at various positions
    print(f"\nTracing scenarios when t_prec = ns_prev (one ULP below next_save):")

    # Scenario: t_prec = ns_prev
    t_prec_test = precision(ns_prev)
    end_of_step = t_prec_test + dt_raw

    print(f"  t_prec = {float(t_prec_test):.20g}")
    print(f"  end_of_step = {float(end_of_step):.15g}")
    print(f"  end_of_step >= next_save: {float(end_of_step) >= ns_val}")

    # dt_eff calculation
    next_event = precision(min(float(t_end), ns_val))
    dt_eff = precision(ns_val - float(t_prec_test))

    print(f"  next_event = {float(next_event):.15g}")
    print(f"  dt_eff = next_save - t_prec = {float(dt_eff):.6e}")

    # What happens to t?
    # We need a t (f64) that rounds to t_prec = ns_prev
    # The rounding boundary between ns_prev and ns_val is at their midpoint
    boundary = (ns_prev + ns_val) / 2

    print(f"\n  Rounding boundary (f64 values below this → t_prec=ns_prev):")
    print(f"    {boundary:.20g}")

    # Test t values just below the boundary
    for offset in [-1e-10, -1e-12, -1e-14]:
        t = float64(boundary + offset)
        t_prec_check = precision(t)
        t_proposal = t + float64(dt_eff)
        t_prec_after = precision(t_proposal)

        print(f"\n  t = boundary{offset:+.0e}:")
        print(f"    t (f64) = {float(t):.20g}")
        print(f"    t_prec = {float(t_prec_check):.20g}")
        print(f"    dt_eff = {float(dt_eff):.6e}")
        print(f"    t_proposal = {float(t_proposal):.20g}")
        print(f"    t_prec_after = {float(t_prec_after):.20g}")
        print(f"    t_prec advanced: {float(t_prec_after) > float(t_prec_check)}")
        print(f"    t_proposal > t: {t_proposal > t}")
        print(f"    t_prec_after >= next_save: {float(t_prec_after) >= ns_val}")


def simulate_adaptive_near_boundary():
    """Simulate adaptive stepping near a save boundary where dt can shrink."""
    print("\n" + "=" * 80)
    print("ADAPTIVE STEPPING NEAR SAVE BOUNDARY")
    print("=" * 80)

    precision = float32

    # Build up to save 80
    save_every = precision(0.1)
    next_save = precision(0.0)
    for _ in range(80):
        next_save = precision(next_save + save_every)

    t_end = precision(8.05)

    # Key: start t just before the rounding boundary where t_prec will jump
    ns_val = float(next_save)
    next_f32 = float(np.nextafter(precision(ns_val), precision(np.inf)))
    boundary = (ns_val + next_f32) / 2

    # Start t slightly below the boundary
    t = float64(boundary - 1e-8)

    print(f"\nSetup:")
    print(f"  next_save = {ns_val:.20g}")
    print(f"  next_f32 = {next_f32:.20g}")
    print(f"  boundary = {boundary:.20g}")
    print(f"  Starting t = {float(t):.20g}")
    print(f"  t_end = {float(t_end):.15g}")

    stagnant_counts = int32(0)

    # Simulate with VERY small adaptive steps
    # This mimics what might happen when the controller shrinks dt near a stiff point
    dt_values = [1e-7, 1e-8, 5e-9, 2e-9, 1e-9, 1e-10, 1e-11]

    for step, dt_small in enumerate(dt_values * 5):  # Repeat the pattern
        if step > 30:
            print(f"\n  ... stopping after {step} steps")
            break

        dt_raw = precision(dt_small)
        t_prec = precision(t)
        end_of_step = t_prec + dt_raw

        save_finished = bool(next_save > t_end)
        do_save = bool(end_of_step >= next_save) and not save_finished

        if do_save:
            next_event = precision(min(float(t_end), float(next_save)))
            dt_eff = precision(float(next_event) - float(t_prec))
        else:
            dt_eff = dt_raw

        t_proposal = t + float64(dt_eff)

        # Stagnation check
        if t_proposal == t:
            stagnant_counts += int32(1)
        else:
            stagnant_counts = int32(0)

        print(f"\n  Step {step}, dt_raw={float(dt_raw):.2e}:")
        print(f"    t (f64):        {float(t):.20g}")
        print(f"    t_prec (f32):   {float(t_prec):.20g}")
        print(f"    next_save:      {float(next_save):.20g}")
        print(f"    t_prec vs next_save: {'>' if float(t_prec) > ns_val else '<=' if float(t_prec) <= ns_val else '='}")
        print(f"    do_save:        {do_save}")
        print(f"    dt_eff:         {float(dt_eff):.6e}")
        print(f"    t_proposal:     {float(t_proposal):.20g}")

        is_backward = t_proposal < t
        is_stagnant = t_proposal == t

        if is_backward:
            print(f"    *** BACKWARD: t_proposal < t by {float(t - t_proposal):.6e} ***")
        if is_stagnant:
            print(f"    *** STAGNANT: t_proposal == t ***")
        print(f"    stagnant_counts: {stagnant_counts}")

        if stagnant_counts >= 2:
            print(f"\n*** STAGNATION TRIGGERED ***")
            break

        # Accept step
        old_t = t
        t = t_proposal
        t_prec = precision(t)

        if is_backward:
            print(f"    >>> t moved backward by {float(old_t - t):.6e}")

        if do_save:
            old_ns = next_save
            next_save = precision(next_save + save_every)
            print(f"    >>> SAVE, next_save: {float(old_ns):.15g} -> {float(next_save):.15g}")

        if next_save > t_end:
            print(f"\n*** Loop would finish (next_save > t_end) ***")
            break


def simulate_repeated_rejection_hang():
    """Simulate the ACTUAL hang: repeated step rejection with non-zero dt_eff.

    KEY INSIGHT: The stagnation check resets when t_proposal != t.
    If steps are repeatedly REJECTED but t_proposal != t, stagnant_counts
    keeps resetting to 0 and NEVER triggers the stagnation exit!
    """
    print("\n" + "=" * 80)
    print("REPEATED REJECTION HANG SCENARIO")
    print("=" * 80)
    print("""
The bug: stagnation check uses (t_proposal == t), which resets to 0
when t_proposal differs from t. But if steps are REJECTED, t never
changes even though t_proposal != t. The counter keeps resetting!
""")

    precision = float32

    # Build up to save 80
    save_every = precision(0.1)
    next_save = precision(0.0)
    for _ in range(80):
        next_save = precision(next_save + save_every)

    t_end = precision(8.05)
    ns_val = float(next_save)

    # Start t at a position where dt_eff will be small but non-zero
    # t_prec should be just below next_save
    ns_prev = float(np.nextafter(precision(ns_val), precision(-np.inf)))
    boundary = (ns_prev + ns_val) / 2

    # Start just below the boundary so t_prec = ns_prev
    t = float64(boundary - 1e-10)

    dt_raw = precision(0.01)
    stagnant_counts = int32(0)

    print(f"Setup:")
    print(f"  next_save = {ns_val:.15g}")
    print(f"  ns_prev (one ULP below) = {ns_prev:.15g}")
    print(f"  Starting t = {float(t):.15g}")
    print(f"  Starting t_prec = {float(precision(t)):.15g}")

    # Track t to detect if it ever changes
    initial_t = t

    print(f"\n--- Simulating ALL steps rejected ---")

    for step in range(20):
        t_prec = precision(t)
        end_of_step = t_prec + dt_raw

        save_finished = bool(next_save > t_end)
        do_save = bool(end_of_step >= next_save) and not save_finished

        if do_save:
            next_event = precision(min(float(t_end), float(next_save)))
            dt_eff = precision(float(next_event) - float(t_prec))
        else:
            dt_eff = dt_raw

        t_proposal = t + float64(dt_eff)

        # CURRENT stagnation check (the bug)
        if t_proposal == t:
            stagnant_counts += int32(1)
        else:
            stagnant_counts = int32(0)  # RESETS even though t won't change!

        stagnant = bool(stagnant_counts >= int32(2))

        print(f"\n  Step {step}:")
        print(f"    t (f64):         {float(t):.15g}")
        print(f"    t_prec (f32):    {float(t_prec):.15g}")
        print(f"    dt_eff:          {float(dt_eff):.6e}")
        print(f"    t_proposal:      {float(t_proposal):.15g}")
        print(f"    t_proposal != t: {t_proposal != t}")
        print(f"    stagnant_counts: {stagnant_counts} (RESETS because t_proposal != t!)")

        if stagnant:
            print(f"\n*** STAGNATION TRIGGERED (won't happen with current bug!) ***")
            break

        # SIMULATE: ALL steps rejected (e.g., solver convergence failure)
        accept = False

        # do_save &= accept (line 869)
        do_save_actual = do_save and accept

        # t = selp(accept, t_proposal, t) - t stays the same!
        if accept:
            t = t_proposal
        # else: t unchanged!

        if do_save_actual:
            next_save = precision(next_save + save_every)
            print(f"    >>> SAVE (won't happen - rejected)")

        # Check if t changed
        if t == initial_t:
            print(f"    >>> t UNCHANGED (accept=False), but stagnant_counts reset!")

        if step >= 10:
            print(f"\n*** INFINITE LOOP: t never changes, stagnant never triggers ***")
            print(f"    t has been {float(t):.15g} for {step+1} iterations")
            print(f"    stagnant_counts keeps resetting because t_proposal != t")
            break

    print(f"\n" + "=" * 80)
    print("ROOT CAUSE IDENTIFIED:")
    print("=" * 80)
    print("""
The stagnation check compares t_proposal to t, but when a step is REJECTED:
  - t stays the same (not updated)
  - t_proposal is recalculated and differs from t
  - stagnant_counts resets to 0
  - Loop never exits via stagnation!

FIX: Track consecutive iterations where t doesn't ACTUALLY advance:

  # Before the step:
  t_before = t

  # After accept/reject logic:
  if t == t_before:  # t didn't actually advance
      stagnant_counts += 1
  else:
      stagnant_counts = 0
""")


def trace_backward_step_scenario():
    """Trace exactly what happens when t_prec overshoots next_save."""
    print("\n" + "=" * 80)
    print("BACKWARD STEP SCENARIO - FULL TRACE")
    print("=" * 80)

    precision = float32

    # Build up next_save to save 80
    save_every = precision(0.1)
    next_save = precision(0.0)
    for _ in range(80):
        next_save = precision(next_save + save_every)

    t_end = precision(8.05)

    ns_val = float(next_save)
    ns_next = float(np.nextafter(precision(ns_val), precision(np.inf)))
    boundary = (ns_val + ns_next) / 2

    print(f"\nSetup:")
    print(f"  next_save = {ns_val:.20g}")
    print(f"  next_f32 (overshoot value) = {ns_next:.20g}")
    print(f"  rounding boundary = {boundary:.20g}")
    print(f"  t_end = {float(t_end):.15g}")

    # Start t just ABOVE the boundary (will round to ns_next, overshooting next_save)
    t = float64(boundary + 1e-10)
    dt_raw = precision(0.01)

    print(f"\n  Starting t = {float(t):.20g}")
    print(f"  Initial t_prec = {float(precision(t)):.20g}")
    print(f"  t_prec > next_save: {float(precision(t)) > ns_val}")

    stagnant_counts = int32(0)

    print(f"\n--- Simulation trace ---")

    for step in range(20):
        t_prec = precision(t)

        # Check finish conditions (from ode_loop lines 691-714)
        end_of_step = t_prec + dt_raw
        save_finished = bool(next_save > t_end)
        finished = save_finished

        if finished:
            print(f"\nStep {step}: FINISHED (save_finished={save_finished})")
            break

        do_save = bool(end_of_step >= next_save) and not save_finished

        if do_save:
            next_event = precision(min(float(t_end), float(next_save)))
            dt_eff = precision(float(next_event) - float(t_prec))
        else:
            dt_eff = dt_raw

        t_proposal = t + float64(dt_eff)

        # Stagnation check
        if t_proposal == t:
            stagnant_counts += int32(1)
        else:
            stagnant_counts = int32(0)

        stagnant = bool(stagnant_counts >= int32(2))

        print(f"\nStep {step}:")
        print(f"  t (f64) = {float(t):.20g}")
        print(f"  t_prec = {float(t_prec):.20g}")
        print(f"  next_save = {float(next_save):.20g}")
        print(f"  t_prec vs next_save: {'>' if float(t_prec) > ns_val else '='}")
        print(f"  do_save = {do_save}")
        print(f"  dt_eff = {float(dt_eff):.6e}")
        print(f"  t_proposal = {float(t_proposal):.20g}")
        print(f"  BACKWARD: {t_proposal < t}")
        print(f"  stagnant_counts = {stagnant_counts}")

        if stagnant:
            print(f"\n*** STAGNATION TRIGGERED ***")
            break

        # In actual code: t = selp(accept, t_proposal, t)
        # Assuming accept=True (step succeeded)
        accept = True
        old_t = t
        t = t_proposal

        if t < old_t:
            print(f"  >>> t moved BACKWARD by {float(old_t - t):.6e}")

        t_prec = precision(t)

        # In actual code: do_save &= accept
        # Then save logic
        if do_save and accept:
            old_ns = next_save
            next_save = precision(next_save + save_every)
            print(f"  >>> SAVE happened, next_save: {float(old_ns):.15g} -> {float(next_save):.15g}")

            # Check if we're done now
            if next_save > t_end:
                print(f"\n*** next_save > t_end, loop would finish on next iteration ***")


def summary_of_findings():
    """Print a summary of the timing analysis findings."""
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

    print("""
ANALYSIS RESULTS:

1. NEGATIVE dt_eff SCENARIO:
   - When t (f64) rounds UP to a t_prec (f32) value > next_save
   - dt_eff = next_save - t_prec becomes NEGATIVE
   - t_proposal = t + dt_eff goes BACKWARD
   - Current stagnation check (t_proposal == t) does NOT catch this

2. WHY THE LOOP DOESN'T HANG (in my simulation):
   - Even with backward t, do_save is True
   - Save happens, next_save += save_every (advances by ~0.1)
   - next_save quickly exceeds t_end
   - Loop exits normally

3. POTENTIAL ISSUES:
   a) Negative dt_eff passed to step function:
      - ODE solvers expect positive dt
      - Could cause incorrect results (states move backward)
      - Could cause solver convergence failures in implicit methods

   b) Stagnation check misses backward movement:
      - Current: if t_proposal == t: stagnant_counts += 1
      - Doesn't catch: t_proposal < t (backward)
      - Recommendation: change to t_proposal <= t

4. WHEN HANG COULD OCCUR:
   - If step is REJECTED when dt_eff is tiny/negative
   - Then t doesn't advance, next_save doesn't advance
   - Could retry indefinitely if rejection keeps happening

5. PROPOSED FIXES:
   a) Change stagnation check from == to <=:
      if t_proposal <= t:
          stagnant_counts += 1

   b) Clamp dt_eff to be non-negative:
      dt_eff = precision(max(0, next_event - t_prec))

   c) Skip to next_save when dt_eff would be tiny/negative:
      if dt_eff <= 0:
          # Force t_prec to exactly next_save
          t = float64(next_save)
""")


if __name__ == "__main__":
    # Run the critical test that replicates the hang
    simulate_repeated_rejection_hang()
