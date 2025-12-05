FROM nginx:alpine

# Copy static viewer files
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 8090
CMD ["nginx", "-g", "daemon off;"]