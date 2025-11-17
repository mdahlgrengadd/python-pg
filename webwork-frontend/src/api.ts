import axios from 'axios';
import { ProblemResponse, GradeResponse, GradeRequest } from './types';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const problemsApi = {
  list: async (): Promise<string[]> => {
    const response = await api.get<string[]>('/problems');
    return response.data;
  },

  get: async (problemId: string, seed?: number): Promise<ProblemResponse> => {
    const response = await api.get<ProblemResponse>(`/problems/${problemId}`, {
      params: { seed },
    });
    return response.data;
  },

  grade: async (problemId: string, request: GradeRequest): Promise<GradeResponse> => {
    const response = await api.post<GradeResponse>(`/problems/${problemId}/grade`, request);
    return response.data;
  },
};

export default api;
