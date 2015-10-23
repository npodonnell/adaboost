#!/usr/bin/perl
#
# Noel P. O'Donnell, Thu Oct 22, 2015
# Got this example from this video: https://www.youtube.com/watch?v=c1xnZ_9jdk8
#
# Finally understanding this algorithm after about 8 hours reading
# paper after paper after paper! 

use strict;
use warnings;
use List::Util qw(sum);

my $EXP=2.71828182845904523536028747135266249775724709369995;

# given an object and an ensemble, will classify the object
my $classify=sub{
    my ($object,@ensemble)=@_;
    
    sum (map{ $_->{wl}->($object) * $_->{alpha} }@ensemble);
};

# exponential loss function
my $elf=sub{
    my ($dp,@ensemble)=@_;

    $EXP ** (-$dp->{class} * $classify->($dp->{object},@ensemble));
};

# objects that need to be classified and their class
my @dataset=(
    { object => 0, class => +1, weight => undef },
    { object => 1, class => +1, weight => undef },
    { object => 2, class => +1, weight => undef },
    { object => 3, class => -1, weight => undef },
    { object => 4, class => -1, weight => undef },
    { object => 5, class => -1, weight => undef },
    { object => 6, class => +1, weight => undef },
    { object => 7, class => +1, weight => undef },
    { object => 8, class => +1, weight => undef },
    { object => 9, class => -1, weight => undef }
);

# weak learners
my @weakLearners=(
    # weak learner 1 (splits on 2.5)
    sub{ (shift) < 2.5 ? +1 : -1 },

    # weak learner 2 (splits on 8.5)
    sub{ (shift) < 8.5 ? +1 : -1 },
    
    # weak learner 3 (splits on 5.5)
    sub{ (shift) > 5.5 ? +1 : -1 }
);

# array which holds the chosen weak learners;
# each will a hashref of the form { wl => .. , alpha => .. }
# where wl is a reference to the weak learner and
# alpha is this weak learner's weight.
my @ensemble=();

# Main AdaBoost algorithm
# for this example we'll keep going until all weak learners
# have been added to the ensemble. This is not a proper
# termination rule, you should use something better.
while (1){

    # STEP 1. Set weights
    if (not scalar @ensemble){
        # First iteration - all datapoints equal
        map{ $_->{weight}= 1/(scalar @dataset); }@dataset;
    }else{
        # compute scaling constant Z after setting weights
        # using exponential loss function
        my $Z=sum(map{ $_->{weight}=$elf->($_,@ensemble); $_->{weight} }@dataset);

        # apply scaling factor
        map{ $_->{weight} /= $Z }@dataset;
    }

    # STEP 2. Select a weak learner
    my $chosenLearnerIndex=-1;
    my $minError=1;
    for my $i (0..$#weakLearners){
        
        next if not defined $weakLearners[$i];
        
        my $totError=0;
        for my $j (0..$#dataset){
            if ($weakLearners[$i]->($dataset[$j]->{object}) != $dataset[$j]->{class}){
                # incorrectly classified
                $totError += $dataset[$j]->{weight};
            }
        }

        if ($totError < $minError){
            # found a useful weak learner
            $chosenLearnerIndex=$i;
            $minError=$totError;
        }
    }

    last if $chosenLearnerIndex==-1; # out of learners

    # get a ref to the newly selected learner then undef it
    # in the weakLearners array so it can't be selected again
    my $chosenLearner=$weakLearners[$chosenLearnerIndex];
    $weakLearners[$chosenLearnerIndex]=undef;

    # STEP 3. Compute alpha
    my $alpha=0.5 * log((1-$minError)/$minError);

    # add it to the ensemble
    push @ensemble,{ wl => $chosenLearner, alpha => $alpha};

    print "Alpha=$alpha\n";
}


# Run the boosted classifier on each object in
# the dataset and check if the output is correct
foreach my $dp (@dataset){
    
    my $output=$classify->($dp->{object}, @ensemble);

    # the 'answer' is the sign of the output
    my $answer=($output > 0 or -1);

    print "Object:$dp->{object} Class:$dp->{class} Classifier Output: $output Answer: $answer (".($dp->{class}==$answer ? "Correct":"Incorrect").")\n";
}

