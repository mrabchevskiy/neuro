                                                                                                                              /*
 Copyright Mykola Rabchevskiy 2023.
 Distributed under the Boost Software License, Version 1.0.
 (See http://www.boost.org/LICENSE_1_0.txt)
 ______________________________________________________________________________


 2023.02.22  Initial version

 Note: Visualization is less general than the rest of code. It can be excluded
       by removing "#include <SFML/Graphics.hpp>" statement and draw(..) function.
       SFML library id the only dependence.

       Code use C++20 features.

________________________________________________________________________________________________________________________________
                                                                                                                              */
#ifndef NET_H_INCLUDED
#define NET_H_INCLUDED

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <SFML/Graphics.hpp>

#include <functional>
#include <initializer_list>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <unordered_map>

using Func   = std::function< double( const double& ) >;
using String = std::string;
using Edge   = std::tuple< unsigned, double, double >;   // :origin node index, edge weight and edge delta

std::default_random_engine RANDOM;

unsigned Nf{ 0 };

std::unordered_map< String, Func > FUNC {
                                                                                                                              /*
  Map name of activation function -> to function object:
                                                                                                                              */
  { "arg", +[]( const double& x ){ assert( false ); return x;                       } }, // :pseudo function for input nodes
  { "sgm", +[]( const double& x ){ Nf++;; return 1.0/( 1.0 + std::exp ( -2.0*x ) ); } }, // :sigmoid
  { "out", +[]( const double& x ){ return x;                                        } }  // :for output node
};


class Data {
                                                                                                                              /*
  Class that represents `training` data set.
  Line contains a single sample as inputs values and expected output (last number),
  separated by special symbol (probaly `,` or TAB symbol):
                                                                                                                              */
  unsigned ARITY;     // :network argument dimensionality
  unsigned N;         // :number of samples
  unsigned CAPACITY;  // :size  of data array
  double* D;          // :data

public:

  unsigned size () const { return N;     }
  unsigned arity() const { return ARITY; }

  Data( const char* path, const char& SEP = ',' ): ARITY( 0 ), N{ 0 }, CAPACITY{ 0 }, D{ nullptr }{
    printf( "\n\n Load data from `%s`.. ", path );
    std::string   line;
    std::ifstream src( path ); assert( src.is_open() );
    while( not src.eof() ){
      std::getline( src, line );
      if( line.empty() ) break;
      if( N == 0 ) for( const auto& symbol: line ) if( symbol == SEP ) ARITY++;  // :first line
      N++;
    }
    printf( " Arity %u, %u samples..", ARITY, N );
    src.close();
    src.open( path ); assert( src.is_open() );
    CAPACITY = ( ARITY + 1 )*N;
    D = new double[ CAPACITY ];
    unsigned n{ 0 };
    while( not src.eof() ){
      std::getline( src, line );
      if( line.empty() ) break;
      std::stringstream S( line );
      for( unsigned i = 0; i <= ARITY; i++ ) S >> D[ n++ ];
    }
    assert( n == CAPACITY );
    printf( " [ok]\n\n" );
    src.close();
  }

  Data( const Data& ) = delete;
  Data& operator= ( const Data& ) = delete;

  const double* operator[] ( unsigned i ) const {
    assert( i < CAPACITY );
    return D + ( ARITY + 1 )*i;
  }

 ~Data(){ if( D ) delete[] D; }

};


class Net {
                                                                                                                              /*
  Class that represents whole neural net; single output assumed:
                                                                                                                              */
  class Node{
                                                                                                                              /*
    Nested class that represents neuron as node of the neural net:
                                                                                                                              */
    double   u;        // :node value
    char     f [ 8 ];  // :function name
    char     ID[ 8 ];  // :node ID
    Func     F;        // :function object
    unsigned n;        // :number of edges
    Edge*    edge;     // :array of edges (size = n+1); first item keeps bias.

    friend class Net;

  public:

    void calc( std::vector< Node >& NODE ){
      const auto& [ j_, c_, d_ ] = edge[0];
      u = c_ + d_;
      for( unsigned i = 1; i <= n; i++ ){
        const auto& [ j, c, d ] = edge[i];
        u += ( c + d )*NODE[j].u;
      }
      u = F( u );
    }

    void randomize(  const double& range ){
      std::uniform_real_distribution< double > uniform( -range, range );
      for( unsigned i = 0; i <= n; i++ ){
        const auto& [ j, c, d ] = edge[i];
        edge[i] = std::make_tuple( j, uniform( RANDOM ), 0.0 );
      }
    }

    void modify( const double& range ){
      std::uniform_real_distribution< double > uniform( -range, range );
      for( unsigned i = 0; i <= n; i++ ){
        const auto& [ j, c, d ] = edge[i];
        edge[i] = std::make_tuple( j, c, uniform( RANDOM ) );
      }
    }

    void invert(){
      for( unsigned i = 0; i <= n; i++ ){
        const auto& [ j, c, d ] = edge[i];
        edge[i] = std::make_tuple( j, c, -d );
      }
    }

    void acceptModification(){
      for( unsigned i = 0; i <= n; i++ ){
        const auto& [ j, c, d ] = edge[i];
        edge[i] = std::make_tuple( j, c + d, 0.0 );
      }
    }

    Node( const char* func, const char* id, const double& bias, const std::unordered_map< unsigned, double >& E ):
      u{ 0.0 }, f{ 0 }, ID{ 0 }, F{ FUNC.at( func ) }, n{ unsigned( E.size() ) }, edge{ new Edge[ n+1 ] }
    {
      assert( func                    );
      assert( id                      );
      assert( std::strlen( id   ) < 8 );
      assert( std::strlen( func ) < 8 );
      strcpy( f,  func );
      strcpy( ID, id   );
                                                                                                                              /*
      Convert set of edges into array:
                                                                                                                              */
      unsigned i{0};
      edge[i++] = std::make_tuple( 0, bias, 0.0 );
      for( const auto& [ p, c ]: E ){
        assert( i <= n );
        edge[i++] = std::make_tuple( p, c, 0.0 ); // :zero delta
      }
    }
                                                                                                                              /*
    Copy constructor used when Node added to array on nodes:
                                                                                                                              */
    Node( const Node& X ): u{ X.u }, f{ 0 }, ID{ 0 }, F{ X.F }, n{ X.n }, edge{ nullptr }{
      strcpy( f,  X.f  );
      strcpy( ID, X.ID );
      edge = new Edge[ n + 1 ];
      for( unsigned i = 0; i <= n; i++ ) edge[i] = X.edge[i];
    }

    Node& operator= (const Node& ) = delete;

    unsigned size() const { return n; }

   ~Node(){ if( edge ) delete[] edge; }

  };//class Node


  const String        TITLE;  // :path to definition file
  unsigned            ARITY;  // :number of inputs
  unsigned            RANGE;  // :number of edges
  std::vector< Node > NODE;   // :array  of nodes

public:

  String   title() const { return TITLE;       }  // :path to definition file
  unsigned size () const { return NODE.size(); }  // :number of nodes
  unsigned arity() const { return ARITY;       }  // :number of input nodes
  unsigned range() const { return RANGE;       }  // :number of edges ~ number of parameters

  Net( const char* path ): TITLE( path ), ARITY{ 0 }, RANGE{ 0 }, NODE{} {

    std::unordered_map< String,   unsigned > M; // :maps node ID -> index in the node[] array
    std::unordered_map< unsigned, double   > E; // :edges as map node index -> edge coefficient

    constexpr char SEP{ ';' };

    char  f[ 16 ];        // :function name
    char  n[ 16 ];        // :currernt node ID
    char  p[ 16 ];        // :parent   node ID
    char* r{ nullptr };   // :node ID after removing possible leading spaces

    char   symbol{ ' ' }; // :parsing symbol
    double bias  { 0.0 }; // :node bias

    FILE*    src = fopen( path, "r" ); assert( src );
    unsigned i   = 0;   // :nodes counter
    for(;;){
                                                                                                                              /*
      Read node activation function name (up to first space):
                                                                                                                              */
      if( EOF == std::fscanf( src, "%15s", f ) ) break;

      printf( "\n %04u  %-3s ", ( ++i ), f ); fflush( stdout );
                                                                                                                              /*
      Read next symbol that must be space:
                                                                                                                              */
      assert( EOF != std::fscanf( src, "%c",  &symbol ) );
      assert( symbol == ' ' );
      if( strcmp( f, "arg" ) == 0 ) ARITY++;
      assert( FUNC.contains( f ) ); // :check the function name
                                                                                                                              /*
      Read node definition as sequence of node ID, symbol ':', and weight/bias value:
                                                                                                                              */
      E.clear();
      bool first{ true }; // :flag meand `first node in sequence`
      for(;;){
        double c{ 0.0 };
                                                                                                                              /*
        Read node ID up to by symbol `:`:
                                                                                                                              */
        assert( EOF != std::fscanf( src, "%15[^:]", p ) );
        r = p; while( *r == ' ' ) r++; // :remove leading spaces
        printf( "%s", r );
                                                                                                                              /*
        Read `:` symbol:
                                                                                                                              */
        assert( EOF != std::fscanf( src, "%c", &symbol ) );
        printf( "%c", symbol ); fflush( stdout ); assert( symbol == ':' );
                                                                                                                              /*
        Read edge weights (or bias in case of first one):
                                                                                                                              */
        assert( EOF != std::fscanf( src, "%lE", &c ) );
        printf( " %E", c ); fflush( stdout );
                                                                                                                              /*
        Read next symbol; acceptable {,;} only:
                                                                                                                              */
        assert( EOF != std::fscanf( src, "%c", &symbol ) );
        printf( "%c ", symbol ); fflush( stdout );
        if( ( symbol != ';' ) and ( symbol != ',' ) ){
          printf( "\n\n Expected [:;] but found `%c`\n\n", symbol ); fflush( stdout ); assert( false );
        }

        if( first ){ // Node ID and bias:
          assert( strlen( r ) < 15 );
          strcpy( n, r ), bias = c;
          assert( not M.contains( n ) ); // :check for node duplication
        } else { // Parent node ID and weight (coefficient):
          if( not M.contains( r ) ){
            printf( "\n\n Invalid edge to unknown node: `%s`\n\n", r );
            fflush( stdout ); assert( false );
          }
          const unsigned i{ M[r] };      // :index referred to parent node;
          assert( not E.contains( i ) ); // :check for edge duplicates
          E[i] = c;
        }
        if( symbol == SEP ) break; // :node sequence terminated
        first = false;
      }//forever
                                                                                                                              /*
      Add node into neural network graph:
                                                                                                                              */
      M[n] = size();
      NODE.emplace_back( f, n, bias, E );

    }//forever
    fclose( src );
    for( const auto& node: NODE ) RANGE += node.size();
                                                                                                                              /*
    Check:
      - a few first nodes should be `arg` ones with empty `arg` list;
      - the last node should be `out`;
      - the rest set of nodes should have non-emplty array of edges and type distinct from {`arg`,`out`}
                                                                                                                              */
    assert( strcmp( NODE.back().f, "out" ) == 0 );
    for( unsigned i = 0; i < size() - 1; i++ ){
      if( i < ARITY ){
        assert( strcmp( NODE[i].f, "arg" ) == 0 );
      } else {
        assert( strcmp( NODE[i].f, "arg" ) != 0 );
        assert( strcmp( NODE[i].f, "out" ) != 0 );
        assert( NODE[i].n > 0                   );
        assert( NODE[i].edge                    );
      }
    }//for i

  }//Net constructor

  void randomize( const double& range ){ for( auto& node: NODE ) node.randomize( range );   }
  void modify   ( const double& range ){ for( auto& node: NODE ) node.modify( range );      }
  void invert()                        { for( auto& node: NODE ) node.invert();             }
  void acceptModification()            { for( auto& node: NODE ) node.acceptModification(); }

  void save( const char* path = nullptr, const bool& print=true ) const {
    if( print ) printf( "\n Save as `%s`.. ", path );
    FILE* out = fopen( path, "w" ); assert( out );
    for( const auto& node: NODE ){
      const auto& [ n, bias, _ ] = node.edge[0];
      fprintf( out, "%s %s:%E", node.f, node.ID, bias );
      for( unsigned i = 1; i <= node.n; i++ ){
        const auto& [ p, c, d ] = node.edge[i]; // :
        const auto& id{ NODE.at( p ).ID };
        fprintf( out, ", %s:%E", id, c );
      }
      fprintf( out, ";\n" );
    }
    fflush( out ); fclose( out );
    if( print ) printf( "[ok]\n" );
  };

  double operator()( const double* arg ){
    for( unsigned i = 0;     i < ARITY;  i++ ) NODE[i].u = arg[i];
    for( unsigned i = ARITY; i < size(); i++ ) NODE[i].calc( NODE );
    return NODE.back().u;
  }

};//class Net


double meanSquareError( Net& net, const Data& data ){
  const unsigned arity{ net.arity() };
  assert( data.arity() == arity );
  const unsigned n{ data.size() };
  double error2{ 0.0 };
  for( unsigned i = 0; i < n; i++ ){
    const double* sample  { data[i]           };
    const double  expected{ sample[ arity ]   };
    const double  actual  { net( sample )     };
    const double  diff    { expected - actual };
    error2 += diff*diff;
  }
  return std::sqrt( error2/n );
}


std::tuple< double, double > improve(
        Net&      net,             // :assumed to be modified
  const Data&     data,
  const unsigned& attempts,
  const double&   startRange,
  const double&   rangeRedureCoeff = 0.8,
  const double&   minRange         = 1.0e-8,
  const unsigned& retry            = 32
){
  double range = startRange;
  printf( "\n\n Improve %s ..", net.title().c_str() );
  double best{ meanSquareError( net, data ) };
  unsigned j{ 0 };
  for( unsigned i = 0; i < attempts; i++ ){
    net.modify( range );
    double Rsq = meanSquareError( net, data );
    if( Rsq < best ){
      best = Rsq;
      net.acceptModification();
      j = 0;
    } else {
      net.invert();
      Rsq = meanSquareError( net, data );
      if( Rsq < best ){
        best = Rsq;
        net.acceptModification();
        j = 0;
      } else {
        if( ++j > retry ){
          range *= rangeRedureCoeff;
          printf( "\n %7u range %12E  Rsq %12E", i, range, best );
          if( range < minRange ) break;
        }
      }
    }
  }
  return std::make_tuple( range, best );
}


void draw( const char* path, Net& net, const Data& data, const double SCALE = 0.01 ){
                                                                                                                              /*
  Image composition:
                                                                                                                              */
  constexpr unsigned SIZE { 400    };
  constexpr unsigned HALF { SIZE/2 };
                                                                                                                              /*
  Values for each image pixel:
                                                                                                                              */
  double H[ SIZE ][ SIZE ];
  double Hmin  {  1.0e3   };
  double Hmax  { -1.0e3   };
  for( unsigned u = 0; u < SIZE; u++ ){                                           // :pixel x-coord
    for( unsigned v = 0; v < SIZE; v++ ){                                         // :pixel y-coord
      double arg[2]{ SCALE*( HALF - double( u ) ), SCALE*( double( v ) - HALF ) };  // :arg ~ { x, y }
      const double h{ net( arg ) };
      H[u][v] = h;
      if( h > Hmax ) Hmax = h;
      if( h < Hmin ) Hmin = h;
    }
  }
  printf( "\n Value range: %.3f .. %.3f", Hmin, Hmax );
                                                                                                                              /*
  Scaling values to range [ 0, 250 ]:
                                                                                                                              */
  constexpr double MAX_BRIGHTNESS{ 250.0 };
  const double factor{ MAX_BRIGHTNESS/( Hmax - Hmin ) };
  for( unsigned u = 0; u < SIZE; u++ ) for( unsigned v = 0; v < SIZE; v++ ) H[u][v] = factor*( H[u][v] - Hmin );
                                                                                                                              /*
  Original dataset parameter:
                                                                                                                              */
  constexpr double R2 = 8.0 / M_PI;  // :radius^2
  printf( "\n\n R2 = %.3f", R2 );
                                                                                                                              /*
  Make image:
                                                                                                                              */
  sf::Image image;
  image.create( 2*SIZE, SIZE );
  const sf::Color C0{ sf::Color{   80,  80,  80 } };
  const sf::Color C1{ sf::Color{  160, 160, 160 } };
  for( unsigned u = 0; u < 400; u++ ){
    for( unsigned v = 0; v < 400; v++ ){
      const uint8_t h = H[u][v];
      image.setPixel( u, v, sf::Color{ h, h, h }  );
      double x = SCALE*( HALF - double( u ) );
      double y = SCALE*( double( v ) - HALF );
      bool out = x*x + y*y > R2;
      image.setPixel( u + SIZE, v, out ? C1 : C0 );
    }
  }

  auto pix = [&]( const unsigned& x, const unsigned& y, const sf::Color& color ){
    if( x < 2*SIZE and y < SIZE ) image.setPixel( x, y, color );
  };

  for( unsigned i = 0; i < data.size(); i++ ){
    const double* sample = data[i];
    double  x = sample[0];
    double  y = sample[1];
    double  f = sample[2];
    uint8_t g = MAX_BRIGHTNESS*f;
    int u = x/SCALE + HALF;
    int v = HALF - y/SCALE;
    sf::Color color{ g, g, g };
    pix( u + SIZE-1, v-1, color );
    pix( u + SIZE,   v-1, color );
    pix( u + SIZE+1, v-1, color );
    pix( u + SIZE-1, v,   color );
    pix( u + SIZE,   v,   color );
    pix( u + SIZE+1, v,   color );
    pix( u + SIZE-1, v+1, color );
    pix( u + SIZE,   v+1, color );
    pix( u + SIZE+1, v+1, color );
  }
  image.saveToFile( path );
}

#endif // NET_H_INCLUDED
