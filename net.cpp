                                                                                                                              /*
 Copyright Mykola Rabchevskiy 2023.
 Distributed under the Boost Software License, Version 1.0.
 (See http://www.boost.org/LICENSE_1_0.txt)
 ______________________________________________________________________________


 2023.02.22  Initial version

 Note: Code uses C++20 features.

________________________________________________________________________________________________________________________________
                                                                                                                              */
#include "net.h"

int main(){

  Data data( "samples.csv", '\t' );

  {
                                                                                                                              /*
    First phase - random search:
                                                                                                                              */
    Net net( "net.txt" );
    printf( "\n\n Net `%s`",     net.title().c_str() );
    printf(   "\n   Arity: %6u", net.arity()         );
    printf(   "\n   Size:  %6u", net.size ()         );
    printf(   "\n   Range: %6u", net.range()         );
    printf( "\n\n Random search phase" ); fflush( stdout );
    constexpr unsigned ATTEMPTS{ 100000 };
    unsigned updated{ 0 };
    double good{ meanSquareError( net, data ) };
    for( unsigned i = 0; i < ATTEMPTS; i++ ){
      net.randomize( 1.0 );
      double Rsq = meanSquareError( net, data );
      if( Rsq < good ){
        net.save( "netA.txt", false );
        good = Rsq;
        updated++;
        printf( "\n   %3u %6u Rsq: %E", updated, i, good );
      }
    }
    printf( "\n   updated %u times", updated );
    if( updated == 0 ){
      printf( "\n\n   Failure: can`t find an y good sulution.\n" );
      return 13;
    }
    printf( ", Rsq: %E", meanSquareError( net, data ) );
    draw( "figA.png", net, data );
  }

  {
                                                                                                                              /*
    Second phase - improvement:
                                                                                                                              */
    printf( "\n\n Improvement phase" );
    Net net( "netA.txt" );
    constexpr unsigned ATTEMPTS          { 100000 };
    constexpr double   START_RANDOM_RANGE{ 1.0    };
    const auto& [ delta, Rsq ] = improve( net, data, ATTEMPTS, START_RANDOM_RANGE );
    printf( "\n\n   Rsq: %E, final delta: %E", Rsq, delta );
    net.save( "netB.txt" );
    draw( "figB.png", net, data );
    printf( "\n\n Finish\n" );
  }
  return 0;
}
