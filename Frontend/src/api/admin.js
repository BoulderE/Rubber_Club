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
  const res = await fetch(`${base}/api/admin/dashboard`, {
    headers: { 'X-Admin-Pin': token }
  })
  return res.json()
}

export async function getUsers(token) {
  const res = await fetch(`${base}/api/admin/users`, {
    method: 'GET',
    headers: { 
      'Content-Type': 'application/json',
      'X-Admin-Pin': token }
  })
  return res.json()
}

export async function createUser(token, data) {
  const res = await fetch(`${base}/api/admin/users`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Admin-Pin': token
    },
    body: JSON.stringify(data)
  })
  const json = await res.json()
  
  if (!res.ok) {
    return { success: false, error: json.error || 'Failed to create user' }
  }
  
  return { success: true, user: json.user }
}

export async function updateUser(token, userId, data) {
  const res = await fetch(`${base}/api/admin/users/${userId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      'X-Admin-Pin': token
    },
    body: JSON.stringify(data)
  })
  const json = await res.json()
  
  if (!res.ok) {
    return { success: false, error: json.error || 'Failed to update user' }
  }
  
  return { success: true, user: json.user }
}

export async function deleteUser(token, userId) {
  const res = await fetch(`${base}/api/admin/users/${userId}`, {
    method: 'DELETE',
    headers: { 'X-Admin-Pin': token }
  })

  if (res.ok) {
    return { success: true }
  }
  
  try {
    const data = await res.json()
    return { success: false, error: data.message || 'Failed to delete user' }
  } catch {
    return { success: false, error: 'Failed to delete user' }
  }
}

export async function getUserHistory(token, userId) {
  const res = await fetch(`${base}/api/admin/users/${userId}/history`, {
    headers: { 'X-Admin-Pin': token }
  })
  return res.json()
}

export async function assignExercise(token, data) {
  const res = await fetch(`${base}/api/admin/assign`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Admin-Pin': token
    },
    body: JSON.stringify(data)
  })
    const json = await res.json()
  
  if (!res.ok) {
    return { success: false, error: json.error || 'Failed to assign' }
  }

  return {success: true, assignment: json.assignment }
}

export async function getAssignments(token, userId = null) {
  let url = `${base}/api/admin/assignments`
  if (userId) url += `?user_id=${userId}`
  const res = await fetch(url, {
    headers: { 'X-Admin-Pin': token }
  })
  return res.json()
}

export async function updateAssignment(token, assignmentId, data) {
  const res = await fetch(`${base}/api/admin/assignments/${assignmentId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      'X-Admin-Pin': token
    },
    body: JSON.stringify(data)
  })
  return res.json()
}

export async function deleteAssignment(token, assignmentId) {
  const res = await fetch(`${base}/api/admin/assignments/${assignmentId}`, {
    method: 'DELETE',
    headers: { 'X-Admin-Pin': token }
  })
  if (res.ok) {
    return { success: true }
  }
  
  try {
    const data = await res.json()
    return { success: false, error: data.message || 'Failed to delete' }
  } catch {
    return { success: false, error: 'Failed to delete' }
  }
}