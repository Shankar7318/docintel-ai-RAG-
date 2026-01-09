# Use Node 20 Alpine image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files for dependency installation
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy all source code
COPY . .

# Build the application for production (optional)
ARG VITE_API_URL=http://backend:8000
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# Expose port for dev server or preview
EXPOSE 5173

# Start Vite dev server (or use `npm run preview` for production)
CMD ["npm", "run", "dev"]