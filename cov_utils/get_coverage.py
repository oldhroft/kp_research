import os

from coverage import coverage
import anybadge

if __name__ == '__main__':

    cov = coverage()
    cov.load()
    total_coverage = cov.report()

    print(f'Code coverage = {total_coverage: .1f}%')
    thresholds = {
        50: 'red',
        70: 'orange',
        90: 'yellow',
        100: 'green'
    }

    badge = anybadge.Badge(
        'coverage', round(total_coverage, 3), thresholds=thresholds,
        value_format='%d%%')

    coverage_path = './cov_utils/coverage.svg'
    if os.path.exists(coverage_path):
        os.remove(coverage_path)
    badge.write_badge(coverage_path)