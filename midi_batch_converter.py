import os
import pretty_midi
import shared_utils

SRC_PATH = 'resources/pop909_complex'
MELODY_PATH = 'resources/pop909_melody'
ACCO_PATH = 'resources/pop909_acco'


def scan(path):
    for filename in os.listdir(path):
        yield filename


def who_is(lst, name):
    for i in lst:
        if i.name == name:
            return i


def convert(filename):
    print('Starting to process MIDI', filename)
    src = pretty_midi.PrettyMIDI(os.path.join(SRC_PATH, filename))
    tempo = src.get_tempo_changes()[1][0]
    print('Original tempo =', tempo)
    melody = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))
    piano.notes.extend(who_is(src.instruments, 'MELODY').notes)
    melody.instruments.append(piano)
    melody.write(os.path.join(MELODY_PATH, filename))
    acco = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))
    piano.notes.extend(who_is(src.instruments, 'PIANO').notes)
    piano.notes.extend(who_is(src.instruments, 'BRIDGE').notes)
    acco.instruments.append(piano)
    acco.write(os.path.join(ACCO_PATH, filename))


def main():
    shared_utils.check_dir(MELODY_PATH)
    shared_utils.check_dir(ACCO_PATH)
    for filename in scan(SRC_PATH):
        convert(filename)


if __name__ == '__main__':
    main()
