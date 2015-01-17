use strict;

my $protofile = shift;
my $monofile = shift;

open(PROTO, $protofile);
my @proto = <PROTO>; # read whole proto
for (1..4) {      # remove first 4 lines (need <BEGINHMM> .. <ENDHMM>
	shift @proto; 
}

# read the monophone file
open(MONO, $monofile);
my $i = 0;
while (my $phone = <MONO>) { 
	chomp $phone;
	print "~h \"$phone\"\n";
	print @proto;
}
close(MONO);
