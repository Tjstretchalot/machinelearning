"""Launches the server and connects a randombot to it. Waits for user input, then connects
the deep1 bot to the gui
"""

import argparse
import secrets
import subprocess
import time

def main():
    """Main entry to the deep1 eval program"""
    parser = argparse.ArgumentParser(
        description='Evaluates the deep1 bot by launching a server and connecting it')
    parser.add_argument('--py3', action='store_true', help='changes executable to python3')
    parser.add_argument('--bot', type=str, default='or_reinforce.deep.deep1.deep1',
                        help='the module and callable of the bot to evaluate')
    parser.add_argument('--gamestart', type=str,
                        default='optimax_rogue.logic.worldgen.TogetherGameStartGenerator',
                        help='Decides how the game is initialized')
    parser.add_argument('ip', type=str, help='the ip for everything to connect to.')
    parser.add_argument('port', type=int, help='the port for the server to be hosted on')
    parser.add_argument('gui_port', type=int, help='the GUI port to connect on')
    args = parser.parse_args()
    _run(args)

def _run(args):
    executable = 'python3' if args.py3 else 'python'

    secret1 = secrets.token_hex(2)
    secret2 = secrets.token_hex(2)

    print(f'---YOUR SECRET IS: \'{secret2}\'---')

    procs = []
    procs.append(subprocess.Popen(
        [executable, '-m', 'optimax_rogue.server.main', '-hn', args.ip,
         '-p', str(args.port), '-t', '0.016', '--dsunused', secret1, secret2,
         '--gamestart', args.gamestart]
    ))

    time.sleep(1)

    procs.append(subprocess.Popen(
        [executable, '-m', 'optimax_rogue_bots.main', '-tr', '0.016',
         args.ip, str(args.port), 'optimax_rogue_bots.randombot.RandomBot', secret1]
    ))

    time.sleep(1)

    input('--press enter to connect gui bot--')

    procs.append(subprocess.Popen(
        [executable, '-m', 'optimax_rogue_bots.gui.main', '-tr', '0.016',
         args.ip, str(args.gui_port), args.bot]
    ))

    print('--waiting--')
    for proc in procs:
        while True:
            if proc.poll():
                break
            time.sleep(1)


    print('--shutting down--')


if __name__ == '__main__':
    main()
