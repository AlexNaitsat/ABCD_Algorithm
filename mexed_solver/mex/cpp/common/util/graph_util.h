// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il (Alexander Naitsat)

#pragma once

#include<iostream> 
#include <list> 
#include <vector> 

#include "boost/graph/adjacency_list.hpp"
#include "boost/property_map/shared_array_property_map.hpp" //this should be included from smallest_last_ordering.hpp
#include <boost/graph/smallest_last_ordering.hpp>
#include <boost/graph/sequential_vertex_coloring.hpp>

using namespace std;
using namespace boost;
namespace util {
	class PartitionGraph
	{
		int N;
		int block_edges_num;
		list<int> *adj, *adj_mesh;
		set<int> *block_edges;

		void DFSUtil(int v, bool visited[]);
		void DFSLabelComponent(int v, bool visited[]);
		void DFSLabelComponentIterative(int v, bool visited[]);
	public:
		int* component; 
		int component_num;    
		int GetComponentNumber() { return component_num; };

		PartitionGraph(int V)
		{
			this->N = V;
			adj = new list<int>[V];
			adj_mesh = new list<int>[V];
			component = new int[V];
			block_edges = NULL;
			block_edges_num = 0;
		};
		~PartitionGraph(){
			if (component) 
				delete[] component;
			if (adj)
				delete[] adj;
			if (adj_mesh)
				delete[] adj_mesh;

		}
		void addMeshEdge(int v, int w);
		void addEdge(int v, int w);
		void LabelConnectedComponents();
		void ComputeBlockGraph();
		void ComputeFreeBlockGraph(const vector<bool>& is_block_fixed);

		int MinimalColoring(vector< vector<int> >& v_same_color);
		int GetVertexComponent(int vi) {
			return(component[vi]);
		};
	};

	int PartitionToBlocks(int vertex_num, int tri_num, int mesh_edges_num, int graph_edges_num,
							const std::vector<std::vector<int>> &vert_simplices,
							const std::vector<std::vector<int>> &vert_neighbors,
						    const double* vv_mesh_edges, const double* vv_graph_edges,
							const std::vector<bool>& is_fixed_vert,
							std::vector<std::set<int>>&  element_blocks,
							std::vector<std::vector<int>>&  free_vertex_blocks,
							std::vector<std::set<int>>&  bnd_vertex_blocks,
							double* vertex_block_index_ptr,						
							std::vector<std::vector<int>>&  blocks_by_color,
							bool run_graph_coloring =false
	);

						   
}