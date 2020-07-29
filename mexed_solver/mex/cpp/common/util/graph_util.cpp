// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il (Alexander Naitsat)

#include "stdafx.h"

#include <iostream>
#include <list>
#include <stack>
#include "common/util/graph_util.h"
#include "data_io/data_io_utils.h"
using namespace std;

namespace util {
	void PartitionGraph::LabelConnectedComponents() {
		component_num = 0;
		for (int i = 0; i < N; i++)	component[i] = -1;
		bool *visited = new bool[N];
		for (int v = 0; v < N; v++)
			visited[v] = false;

		for (int v = 0; v < N; v++)
		{
			if (visited[v] == false)
			{
				component[v] = component_num;
				DFSLabelComponentIterative(v, visited);
				component_num++;
			}
		}
	}

	void PartitionGraph::DFSUtil(int v, bool visited[])
	{
		visited[v] = true;

		list<int>::iterator i;
		for (i = adj[v].begin(); i != adj[v].end(); ++i)
			if (!visited[*i])
				DFSUtil(*i, visited);
	}

	void PartitionGraph::DFSLabelComponent(int v, bool visited[])
	{
		visited[v] = true;

		list<int>::iterator i;
		for (i = adj[v].begin(); i != adj[v].end(); ++i)
			if (!visited[*i]) {
				component[*i] = component_num;
				DFSLabelComponent(*i, visited);
			}
	}


	void PartitionGraph::DFSLabelComponentIterative(int v0, bool visited[])
	{
		std::stack<int> stack;
		stack.push(v0);

		while (!stack.empty())
		{
			auto v = stack.top();
			stack.pop();
			visited[v] = true;

			for (auto i: adj[v])
				if (!visited[i]) {
					component[i] = component_num;
					stack.push(i);
				}
		}
	}
		


	void PartitionGraph::ComputeBlockGraph()
	{
		LabelConnectedComponents();
		block_edges = new set<int>[component_num];
		block_edges_num = 0;

		for (int v = 0; v < N; v++) {
			int v_comp = component[v];
			for (auto u : adj_mesh[v]) {
				int u_comp = component[u];
				int min_comp = std::min(v_comp, u_comp);
				int max_comp = std::max(v_comp, u_comp);

				auto& min_comp_edges = block_edges[min_comp];
				auto status = min_comp_edges.insert(max_comp);
				if (status.second)
					block_edges_num++;
			}
		}
	}



	void PartitionGraph::ComputeFreeBlockGraph(const vector<bool>& is_block_fixed )
	{
		block_edges = new set<int>[component_num];
		block_edges_num = 0;

		for (int v = 0; v < N; v++) {
			int v_comp = component[v];
			if (is_block_fixed[v_comp])
				continue;
			for (auto u : adj_mesh[v]) {
				int u_comp = component[u];
				if (is_block_fixed[u_comp])
					continue;
				int min_comp = std::min(v_comp, u_comp);
				int max_comp = std::max(v_comp, u_comp);

				auto& min_comp_edges = block_edges[min_comp];
				auto status = min_comp_edges.insert(max_comp);
				if (status.second)
					block_edges_num++;
			}
		}
	}

	void PartitionGraph::addMeshEdge(int v, int w)
	{
		adj_mesh[std::min(v,w)].push_back(std::max(v, w));
	}

	void PartitionGraph::addEdge(int v, int w)
	{
		adj[v].push_back(w);
		adj[w].push_back(v);
	}

	int PartitionGraph::MinimalColoring(std::vector< std::vector<int> >& v_same_color) {
		typedef adjacency_list<listS, vecS, undirectedS> Graph;
		typedef graph_traits<Graph>::vertex_descriptor vertex_descriptor;
		typedef graph_traits<Graph>::vertices_size_type vertices_size_type;
		typedef property_map<Graph, vertex_index_t>::const_type vertex_index_map;

		if (!block_edges_num) {
			v_same_color.clear();
			v_same_color.resize(1, std::vector<int>{});
			v_same_color[0].push_back(0);
			return 1;
		}

		Graph g;
		for (int i = 0; i < component_num; ++i)
		{
			add_vertex(g);
		}

		for (int bi = 0; bi < component_num; bi++)
		{
			for (auto bi_neighbor : block_edges[bi]) {
				if (bi < bi_neighbor) {
					add_edge(bi, bi_neighbor, g);
				}
			}
		}

		boost::vector_property_map<vertex_descriptor> order;
		smallest_last_vertex_ordering(g, order);

		std::vector<vertices_size_type> color_vec(num_vertices(g));
		iterator_property_map<vertices_size_type*, vertex_index_map>
		color(&color_vec.front(), get(vertex_index, g));
		vertices_size_type num_colors = sequential_vertex_coloring(g, order, color);

		v_same_color.clear(); v_same_color.resize(num_colors, std::vector<int>{});
		for (int i = 0; i < component_num; ++i)
		{
			v_same_color[color_vec[i]].push_back(i);
		}

		return num_colors;
	}


	int PartitionToBlocks(int vertex_num, int tri_num, int mesh_edges_num, int graph_edges_num,
							const std::vector<std::vector<int>> &vert_simplices,
							const std::vector<std::vector<int>> &vert_neighbors,
							const double* vv_mesh_edges, const double* vv_graph_edges,
							const std::vector<bool>& is_fixed_vert,
							std::vector<std::set<int>>&  element_blocks,
							std::vector<std::vector<int>>&  free_vertex_blocks,
							std::vector<std::set<int>>&  bnd_vertex_blocks,
							double* vertex_block_index_ptr,
							std::vector<std::vector<int>>& blocks_by_color,
							bool run_graph_coloring)
	{
		using namespace data_io;

		util::PartitionGraph graphMesh(vertex_num);

		for (int i = 0; i < mesh_edges_num; i++) {
			graphMesh.addMeshEdge(vv_mesh_edges[i], vv_mesh_edges[i + mesh_edges_num]);
		}

		for (int i = 0; i < graph_edges_num; i++) {
			graphMesh.addEdge(vv_graph_edges[i], vv_graph_edges[i + graph_edges_num]);
		}

		graphMesh.LabelConnectedComponents();

		int block_num = graphMesh.GetComponentNumber();
		free_vertex_blocks.resize(block_num);
		bnd_vertex_blocks.resize(block_num);
		element_blocks.resize(block_num);

		std::vector<bool> is_fixed_block(block_num, true);

		for (int v = 0; v < vertex_num; v++) {
			int b = graphMesh.GetVertexComponent(v);

			if (!is_fixed_vert[v]) {
				free_vertex_blocks[b].push_back(v);
				is_fixed_block[b] = false;
				for (auto t : vert_simplices[v])
					auto status = element_blocks[b].insert(t);

				for (auto v_neigh : vert_neighbors[v]) {
					int v_neigh_block = graphMesh.GetVertexComponent(v_neigh);
					if (v_neigh_block != b)
						bnd_vertex_blocks[b].insert(v_neigh);
				}
			}
			else {
				bnd_vertex_blocks[b].insert(v);
			}
		}

		int color_num = 0;
		if (run_graph_coloring) {
			graphMesh.ComputeFreeBlockGraph(is_fixed_block);
			color_num=graphMesh.MinimalColoring(blocks_by_color);
		}
		else {
			for (int b = 0;b < block_num; b++) {
				std::vector<int> b_vec(1, b);
				blocks_by_color.emplace_back(b_vec);
			}
			color_num = block_num;
		}

		for (size_t i = 0; i < vertex_num; i++)
			vertex_block_index_ptr[i] = graphMesh.GetVertexComponent(i);
		
		return color_num;
	}


}
