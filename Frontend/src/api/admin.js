import { getApiBase } from './base'

const base = getApiBase()

export async function adminLogin(pin) {
  const res = await fetch(`${base}/api/admin/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pin })
  })
  return res.json()
}

export async function getStats(token) {
  const res = await fetch(`${base}/api/admin/stats`, {
    headers: { 'Authorization': `Bearer ${token}` }
  })
  return res.json()
}

export async function getUsers(token) {
  const res = await fetch(`${base}/api/admin/users`, {
    headers: { 'Authorization': `Bearer ${token}` }
  })
  return res.json()
}

export async function getUserHistory(token, userId) {
  const res = await fetch(`${base}/api/admin/users/${userId}/history`, {
    headers: { 'Authorization': `Bearer ${token}` }
  })
  return res.json()
}

export async function assignExercise(token, data) {
  const res = await fetch(`${base}/api/admin/assign`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify(data)
  })
  return res.json()
}

export async function getAssignments(token, userId = null) {
  let url = `${base}/api/admin/assignments`
  if (userId) url += `?user_id=${userId}`
  const res = await fetch(url, {
    headers: { 'Authorization': `Bearer ${token}` }
  })
  return res.json()
}

export async function updateAssignment(token, assignmentId, data) {
  const res = await fetch(`${base}/api/admin/assignments/${assignmentId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify(data)
  })
  return res.json()
}

export async function deleteAssignment(token, assignmentId) {
  const res = await fetch(`${base}/api/admin/assignments/${assignmentId}`, {
    method: 'DELETE',
    headers: { 'Authorization': `Bearer ${token}` }
  })
  return res.json()
}