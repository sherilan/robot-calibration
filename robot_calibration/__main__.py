import argparse
import logging
import json
import sys

import numpy as np

import robot_calibration.est as est

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    'ipt',
    nargs='?',
    help=(
        'Input data for estimation. If not provided, then the program will '
        'attempt to read it from standard in. It should be given as a space '
        'separated table of floats where the number of columns should be '
        'equal to 35. The first 16 columns is the world to robot base transform '
        '(as given by the mocap system). The next 16 columns is the robot base '
        'transform (as given by the robot forward kinematics). The last 3 columns '
        'is the xyz coordinates of the tip of the pen.'
    )
)
parser.add_argument('--verbose', '-v', action='store_true', help='Whether to enable logging')
parser.add_argument('--x0', default=0.0, type=float, help='Initial guess for x0 in meters')
parser.add_argument('--y0', default=0.0, type=float, help='Initial guess for y0 in meters')
parser.add_argument('--z0', default=0.0, type=float, help='Initial guess for z0 in meters')
parser.add_argument('--a0', default=0.0, type=float, help='Initial guess for a0 in meters')
parser.add_argument('--p0', default=20.0, type=float, help='Initial guess for p0 in meters')
parser.add_argument('--lr', default=1.0, type=float, help='Optimizer step size')
parser.add_argument('--iters', default=25, type=float, help='Number of optimization steps')
parser.add_argument('--fmt', default='transforms', choices=['transforms', 'params'], help='Output format')
parser.add_argument('--out', help='Output file (standard out if not provided)')


def main(args):

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='[%(levelname)s] %(asctime)s: %(message)s'
    )

    if args.ipt is None:
        logger.info('Reading data from stdin...')
        data = np.loadtxt(sys.stdin)
    else:
        logger.info('Reading data from %s', args.ipt)
        data = np.loadtxt(args.ipt)
    logger.info('Read data with shape: %s', data.shape)

    expected_cols = 4 * 4 + 4 * 4 + 3
    if not data.ndim == 2 or not data.shape[1] == expected_cols:
        logger.error(
            'Data should be given as a ?x%s matrix. Received: %s',
            expected_cols,
            data.shape
        )
        return 1

    Tb = est.to_tensor(data[:, 0:16].reshape(-1, 4, 4))
    Tr = est.to_tensor(data[:, 16:32].reshape(-1, 4, 4))
    pen_xyz = est.to_tensor(data[:, 32:].reshape(-1, 3))

    logger.info('Fitting parameters...')
    estimator, final_loss = est.Estimator.fit(
        data=(Tb, Tr, pen_xyz),
        init=dict(
            x0=args.x0,
            y0=args.y0,
            z0=args.z0,
            a0=args.a0,
            p0=args.p0,
        ),
        lr=args.lr,
        iters=args.iters,
        pbar=args.verbose,
    )
    logger.info('Finished fitting. Final loss: %s', final_loss)
    logger.info('Estimated params: \n%s', estimator.params)
    logger.info('Corresponding transforms: \n%s', estimator.transforms)

    if args.fmt == 'params':
        if args.out is None:
            logger.info('Writing params to stdout...')
            json.dump(estimator.params, sys.stdout)
        else:
            logger.info('Writing params to %s ...', args.out)
            with open(args.out, 'w') as f:
                json.dump(estimator.params, f)
    else:
        transforms = estimator.transforms.reshape(2, 16)
        if args.out is None:
            logger.info('Writing transforms to stdout...')
            np.savetxt(sys.stdout, transforms)
        else:
            logger.info('Writing transforms to %s ...', args.out)
            np.savetxt(args.out, transforms)


    logger.info('Done!')

    return 0





if __name__ == '__main__':
    sys.exit(main(parser.parse_args()))
