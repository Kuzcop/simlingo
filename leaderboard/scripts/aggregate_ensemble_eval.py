#!/usr/bin/env python
#
# Aggregate the per-GPU result shards produced by
# run_simlingo_town13_eval_parallel.sh into one combined summary.
# (Ported verbatim from carla_garage/leaderboard/scripts/aggregate_ensemble_eval.py; the shard
#  format is standard LB2.0, so it self-adds simlingo's own leaderboard + scenario_runner below.)
#
# Scoring matches Dimitri's leaderboard/testbench.ipynb, i.e. the Leaderboard 2.1 infraction model:
#
#     score_penalty  = 1 / (1 + sum_j c_j * #infractions_j)      # LB 2.1 (additive)
#     score_composed = score_route * score_penalty
#
# RC, IP and DS are each an INDEPENDENT mean over routes (DS = mean(RC_i * IP_i) != mean(RC)*mean(IP);
# this matches the CARLA leaderboard reporting). The old LB 2.0 multiplicative penalty (the
# score_penalty already stored in the result JSON by CARLA) is reported side by side for comparison.
#
# This still merges the shard records into one results.json via the leaderboard's own
# StatisticsManager (so the file carries the standard LB2.0 global_record and the monitor's
# "finished" detection keeps working); the LB2.1 numbers are computed here from the per-route
# infraction counts, exactly as testbench.ipynb does. No route XML / CARLA server is needed.
#
# Cross-run averaging (mean +/- SE over run_1..run_N) is done by testbench.ipynb; this script
# summarizes a single run's shards (one testbench "root").
#
# Usage:
#   python3 aggregate_ensemble_eval.py --checkpoints-dir results/ensemble_eval/run_1/checkpoints \
#       -e results/ensemble_eval/run_1/results_merged.json
#   python3 aggregate_ensemble_eval.py -f a.json b.json c.json -e merged.json

import argparse
import glob
import os
import sys

# Make the repo's leaderboard + scenario_runner importable even without PYTHONPATH set (the
# StatisticsManager imports srunner.scenariomanager.traffic_events). carla itself must be importable
# from the active conda env (same env used to run the evaluation).
_LEADERBOARD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_WORK_DIR = os.path.abspath(os.path.join(_LEADERBOARD_ROOT, os.pardir))
for _p in (_LEADERBOARD_ROOT, os.path.join(_WORK_DIR, 'scenario_runner')):
  if _p not in sys.path:
    sys.path.insert(0, _p)

from leaderboard.utils.checkpoint_tools import fetch_dict  # noqa: E402
from leaderboard.utils.statistics_manager import StatisticsManager  # noqa: E402

# LB 2.1 penalty coefficients (score_penalty = 1 / (1 + sum c_j * count_j)). Keep in sync with
# testbench.ipynb PENALTY_COEFFICIENTS. outside_route_lanes is percentage-based -> coefficient 0.
PENALTY_COEFFICIENTS = {
    'collisions_pedestrian': 1.0,
    'collisions_vehicle': 0.70,
    'collisions_layout': 0.60,
    'red_light': 0.40,
    'stop_infraction': 0.25,
    'scenario_timeouts': 0.40,
    'yield_emergency_vehicle_infractions': 0.40,
    'min_speed_infractions': 0.40,
    'outside_route_lanes': 0.0,
}
COLLISION_TYPES = ['collisions_pedestrian', 'collisions_vehicle', 'collisions_layout']


def collect_files(args):
  """Return the list of shard result JSONs, from --checkpoints-dir glob or an explicit -f list."""
  if args.file_paths:
    return sorted(args.file_paths)
  pattern = os.path.join(args.checkpoints_dir, 'results_gpu*_*.json')
  files = sorted(glob.glob(pattern))
  if not files:
    raise SystemExit(f'No shard files matching {pattern}')
  return files


def scan_shards(files):
  """First pass (mirrors merge_statistics.py): count records and expected total across shards."""
  route_ids = []
  total_routes = 0     # records actually present
  total_progress = 0   # routes expected (sum of each shard's progress[1])
  sensors = []
  per_shard = []       # (file, n_records, expected) for reporting
  for f in files:
    data = fetch_dict(f)
    if not data:
      per_shard.append((f, 0, 0))
      continue
    records = data['_checkpoint']['records']
    expected = data['_checkpoint']['progress'][1] if data['_checkpoint'].get('progress') else 0
    route_ids.extend([r['route_id'] for r in records])
    total_routes += len(records)
    total_progress += expected
    per_shard.append((f, len(records), expected))
    if data.get('sensors'):
      if not sensors:
        sensors = data['sensors']
      elif data['sensors'] != sensors:
        raise SystemExit('Stopping: found two shards with different sensor configurations')
  return route_ids, total_routes, total_progress, sensors, per_shard


def merge(files, endpoint, total_routes, total_progress, sensors, complete):
  """Merge shard records via StatisticsManager and (if complete) recompute + validate the global record."""
  sm = StatisticsManager(endpoint, 0)
  for f in files:
    sm.add_file_records(f)
  sm.sort_records()
  sm.save_sensors(sensors)
  sm.save_progress(total_routes, total_progress)
  sm.save_entry_status('Started')
  if complete:
    sm.compute_global_statistics()
    sm.validate_and_write_statistics(True, False)
  else:
    # Keep the written file honest (no fabricated global_record over a partial run).
    sm.write_statistics()
  return sm


def lb21_penalty(infractions):
  """LB 2.1 additive penalty from a per-route infractions dict (name -> list of messages)."""
  weighted = sum(coef * len(infractions.get(name, [])) for name, coef in PENALTY_COEFFICIENTS.items())
  return 1.0 / (1.0 + weighted)


def summarize(records):
  """Per-route old (LB2.0) + new (LB2.1) scores, reduced to means over routes (testbench-style)."""
  n = len(records)
  rc = old_ip = old_ds = new_ip = new_ds = 0.0
  km_driven = total_length_km = 0.0
  coll_totals = {ct: 0 for ct in COLLISION_TYPES}
  inf_totals = {name: 0 for name in PENALTY_COEFFICIENTS}

  for r in records:
    route = r.scores['score_route']
    rlen_km = r.meta['route_length'] / 1000.0
    new_pen = lb21_penalty(r.infractions)

    rc += route / n
    old_ip += r.scores['score_penalty'] / n
    old_ds += r.scores['score_composed'] / n
    new_ip += new_pen / n
    new_ds += (route * new_pen) / n

    km_driven += route / 100.0 * rlen_km
    total_length_km += rlen_km
    for name in inf_totals:
      inf_totals[name] += len(r.infractions.get(name, []))
    for ct in coll_totals:
      coll_totals[ct] += len(r.infractions.get(ct, []))

  return dict(n=n, rc=rc, old_ip=old_ip, old_ds=old_ds, new_ip=new_ip, new_ds=new_ds,
              km_driven=km_driven, total_length_km=total_length_km,
              coll_totals=coll_totals, inf_totals=inf_totals)


def print_summary(records, per_shard, total_routes, total_progress, complete, endpoint):
  s = summarize(records)
  km = s['km_driven']
  print('')
  print('=' * 66)
  print('ENSEMBLE EVAL — AGGREGATED SUMMARY  (LB2.1 scoring, testbench-compatible)')
  print('=' * 66)
  print('Shards:')
  for f, n, exp in per_shard:
    tag = '' if (exp and n == exp) else (f'  [INCOMPLETE: {n}/{exp}]' if exp else '  [empty]')
    print(f'  {os.path.basename(f)}: {n} routes{tag}')
  if complete:
    print(f'Coverage: COMPLETE — {total_routes}/{total_progress} routes across {len(per_shard)} shards')
  else:
    missing = total_progress - total_routes
    print(f'Coverage: PARTIAL — {total_routes}/{total_progress} routes ({missing} missing). '
          f'Means below are over COMPLETED routes only; NO global_record written to the merged file.')
  print('-' * 66)
  print(f'Routes aggregated: {s["n"]}')
  print(f'{"Metric":<28}{"Old (LB2.0)":>18}{"New (LB2.1)":>18}')
  print(f'{"-" * 64}')
  print(f'{"Route Completion (RC)":<28}{s["rc"]:>18.2f}{s["rc"]:>18.2f}')
  print(f'{"Infraction Penalty (IP)":<28}{s["old_ip"]:>18.4f}{s["new_ip"]:>18.4f}')
  print(f'{"Driving Score (DS)":<28}{s["old_ds"]:>18.4f}{s["new_ds"]:>18.4f}')
  print(f'{"-" * 64}')
  print('  Note: DS = mean(RC_i * IP_i), an independent mean over routes (!= RC * IP).')
  print('-' * 66)
  print('Collision rates (per km driven):')
  for ct in COLLISION_TYPES:
    rate = s['coll_totals'][ct] / km if km > 0 else 0.0
    print(f'  {ct:<40} {rate:.4f}/km   (total: {s["coll_totals"][ct]})')
  print('-' * 66)
  print('Infraction totals (count, LB2.1 coefficient c_j):')
  for name, coef in PENALTY_COEFFICIENTS.items():
    tot = s['inf_totals'][name]
    if tot > 0:
      print(f'  {name:<40} count={tot:<6} c_j={coef:.2f}')
  print('-' * 66)
  print(f'Total distance driven: {km:.3f} km')
  print(f'Total route length:    {s["total_length_km"]:.3f} km')
  print(f'Merged results written to: {endpoint}')
  print('=' * 66)


def main():
  ap = argparse.ArgumentParser(description='Aggregate parallel ensemble-eval result shards (LB2.1 scoring).')
  ap.add_argument('--checkpoints-dir', default=None,
                  help='Directory holding results_gpu*_*.json shards (default source of files).')
  ap.add_argument('-f', '--file-paths', nargs='+', default=None,
                  help='Explicit shard files (overrides --checkpoints-dir).')
  ap.add_argument('-e', '--endpoint', default=None,
                  help='Output merged JSON path (default: <checkpoints-dir>/../results_merged.json).')
  args = ap.parse_args()

  if not args.file_paths and not args.checkpoints_dir:
    ap.error('provide --checkpoints-dir or -f/--file-paths')

  files = collect_files(args)
  endpoint = args.endpoint
  if endpoint is None:
    base = args.checkpoints_dir if args.checkpoints_dir else os.path.dirname(files[0])
    endpoint = os.path.join(os.path.dirname(os.path.abspath(base.rstrip('/'))), 'results_merged.json')

  route_ids, total_routes, total_progress, sensors, per_shard = scan_shards(files)
  if total_routes == 0:
    raise SystemExit('No route records found in any shard.')
  if len(set(route_ids)) != len(route_ids):
    dupes = sorted({rid for rid in route_ids if route_ids.count(rid) > 1})
    raise SystemExit(f'Duplicate route ids across shards (overlapping subsets?): {dupes}')

  complete = total_progress != 0 and total_routes == total_progress
  sm = merge(files, endpoint, total_routes, total_progress, sensors, complete)
  records = sm._results.checkpoint.records  # pylint: disable=protected-access
  print_summary(records, per_shard, total_routes, total_progress, complete, endpoint)


if __name__ == '__main__':
  main()
