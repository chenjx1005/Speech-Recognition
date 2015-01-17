#!usr/bin/env python
# -*- coding: utf-8 -*-
import os

#this script is used to generate some text script
def HCopyScript():
	d = 'test'
	audio = os.listdir('audio/'+d)
	for i in range(len(audio)):
		audio[i] = 'audio/%s/%s mfcc/%s/%s.mfcc\n' % (d, audio[i], d, audio[i])
	f = open('analysis/digit.wav2mfcc.scp','w')
	f.writelines(audio)
	f.close()

def HCompv():
	mfcc = os.listdir('mfcc/test')
	for i in range(len(mfcc)):
		mfcc[i] =  'mfcc/test/%s\n' % mfcc[i]
	f = open('mfcc/test.scp','w')
	f.writelines(mfcc)
	f.close()

def add_mlf_08():
	filename=['label/dev.ref.mlf', 'label/train.all.mlf',
			'label/train.nosp.mlf', 'label/train.phones0.mlf']
	outname = ['label/dev.ref.08.mlf', 'label/train.all.08.mlf',
			'label/train.nosp.08.mlf', 'label/train.phones0.08.mlf']
	for i in range(len(filename)):
		r = open(filename[i], 'r')
		w = open(outname[i], 'w')
		for line in r:
			w.write(line.replace('.lab','.08.lab'))
		r.close()
		w.close()

if __name__ == '__main__':
	HCompv()